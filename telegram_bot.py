#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram-бот «AI-сборщик портфеля облигаций».
Тонкая обёртка над tinvest_bond_bot.py. Режим: long polling (для Railway worker).

Переменные окружения (задаются в Railway → Variables):
  TELEGRAM_TOKEN  — токен бота от @BotFather (обязательно)
  TINVEST_TOKEN   — readonly-токен T-Invest (для реальных данных; без него — только /demo)
  ALLOWED_USERS   — список Telegram user id через запятую, кому можно (НАСТОЯТЕЛЬНО задать!)
  MAX_CANDIDATES  — сколько облигаций тянуть из T-Invest (по умолчанию 60)
  INCLUDE_QUAL    — "1" чтобы включать бумаги для квалов (по умолчанию выкл)

Команды:
  /start /help — справка
  /id          — показать свой Telegram id (чтобы вписать в ALLOWED_USERS)
  /demo 1500000 3        — портфели на синтетике (без токена), сравнение 3 профилей
  /p 1500000 3           — реальные данные, сравнение 3 профилей + кнопки
  /p 1500000 3 mod 18    — конкретный профиль (риск mod/cons/agg), реинвест 18%
  или просто сообщение:  1500000 3 mod
"""
import asyncio
import logging
import os
import time

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (Application, CommandHandler, MessageHandler,
                          CallbackQueryHandler, ContextTypes, filters)

from tinvest_bond_bot import (build_portfolio, demo_universe,
                              fetch_universe_from_tinvest, PROFILES,
                              render_portfolio_html, render_compare_html)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

TG_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TINVEST_TOKEN = os.environ.get("TINVEST_TOKEN")
ALLOWED = {int(x) for x in os.environ.get("ALLOWED_USERS", "").replace(" ", "").split(",") if x}
MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "60"))
INCLUDE_QUAL = os.environ.get("INCLUDE_QUAL", "0") == "1"
DEFAULT_REINVEST = 18.0
DEFAULT_TAX = True

_CACHE = {}            # key -> (timestamp, universe)
_CACHE_TTL = 300       # сек — реальная вселенная кешируется на 5 минут

HELP = (
    "<b>AI-сборщик портфеля облигаций</b>\n\n"
    "Пришли параметры — соберу портфель на реальных данных T-Invest.\n\n"
    "<b>Формат:</b> <code>сумма срок [риск] [реинвест%]</code>\n"
    "Примеры:\n"
    "• <code>1500000 3</code> — сравнение 3 профилей\n"
    "• <code>1500000 3 mod</code> — умеренный, полный состав\n"
    "• <code>3000000 5 agg 20</code> — агрессивный, реинвест 20%\n\n"
    "Риск: <code>cons</code> (консерв.), <code>mod</code> (умерен.), <code>agg</code> (агресс.)\n\n"
    "<b>Команды:</b>\n"
    "/demo 1500000 3 — на синтетике, без реальных данных\n"
    "/id — узнать свой Telegram id\n"
    "/help — эта справка\n\n"
    "⚠️ Это не инвестиционная рекомендация. Бот считает план, заявки не выставляет."
)


def is_allowed(uid: int) -> bool:
    return (not ALLOWED) or (uid in ALLOWED)


def parse_params(parts):
    """['1500000','3','mod','18'] -> (amount, term, risk|None, reinvest, ok, err)"""
    if len(parts) < 2:
        return None, None, None, DEFAULT_REINVEST, False, "Нужно минимум: сумма и срок."
    try:
        amount = float(parts[0].replace(" ", "").replace("_", ""))
        term = float(parts[1].replace(",", "."))
    except ValueError:
        return None, None, None, DEFAULT_REINVEST, False, "Сумма и срок должны быть числами."
    if amount < 10000 or amount > 1e9:
        return None, None, None, DEFAULT_REINVEST, False, "Сумма вне разумного диапазона."
    if term < 0.5 or term > 15:
        return None, None, None, DEFAULT_REINVEST, False, "Срок должен быть от 0.5 до 15 лет."
    risk = None
    reinvest = DEFAULT_REINVEST
    for p in parts[2:]:
        pl = p.lower()
        if pl in PROFILES:
            risk = pl
        else:
            try:
                reinvest = float(pl.replace(",", "."))
            except ValueError:
                pass
    return amount, term, risk, reinvest, True, None


def get_universe(term, demo):
    if demo:
        return demo_universe()
    if not TINVEST_TOKEN:
        raise RuntimeError("TINVEST_TOKEN не задан — доступен только /demo.")
    key = (round(term), INCLUDE_QUAL, MAX_CANDIDATES)
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < _CACHE_TTL:
        return hit[1]
    uni = fetch_universe_from_tinvest(TINVEST_TOKEN, term, MAX_CANDIDATES, INCLUDE_QUAL, 4, verbose=False)
    _CACHE[key] = (now, uni)
    return uni


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP, parse_mode=ParseMode.HTML)


async def cmd_id(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    await update.message.reply_text(
        f"Твой Telegram id: <code>{u.id}</code>\n"
        f"Впиши его в переменную ALLOWED_USERS на Railway, чтобы пользоваться ботом.",
        parse_mode=ParseMode.HTML)


async def _run(update: Update, ctx: ContextTypes.DEFAULT_TYPE, parts, demo):
    uid = update.effective_user.id
    if not is_allowed(uid):
        await update.message.reply_text(
            "Доступ закрыт. Узнай свой id командой /id и попроси владельца добавить тебя.")
        return
    amount, term, risk, reinvest, ok, err = parse_params(parts)
    if not ok:
        await update.message.reply_text(f"⚠️ {err}\n\nПодсказка: /help")
        return

    src = "синтетике" if demo else "реальных данных T-Invest"
    msg = await update.message.reply_text(f"⏳ Считаю на {src}… (может занять до минуты)")

    try:
        universe = await asyncio.to_thread(get_universe, term, demo)
    except Exception as e:
        await msg.edit_text(f"❌ Не удалось получить данные: {e}")
        return
    if not universe:
        await msg.edit_text("❌ Не нашёл подходящих бумаг (проверь фильтры/доступ).")
        return

    # кешируем вселенную и параметры для кнопок «развернуть профиль»
    ctx.chat_data["universe"] = universe
    ctx.chat_data["amount"] = amount
    ctx.chat_data["term"] = term
    ctx.chat_data["reinvest"] = reinvest
    ctx.chat_data["demo"] = demo

    if risk:
        res = await asyncio.to_thread(build_portfolio, universe, amount, term, risk)
        await msg.edit_text(render_portfolio_html(res, amount, term, reinvest, DEFAULT_TAX),
                            parse_mode=ParseMode.HTML)
        return

    results = {p: await asyncio.to_thread(build_portfolio, universe, amount, term, p)
               for p in ("cons", "mod", "agg")}
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("Консерв.", callback_data="show|cons"),
        InlineKeyboardButton("Умерен.", callback_data="show|mod"),
        InlineKeyboardButton("Агресс.", callback_data="show|agg"),
    ]])
    await msg.edit_text(render_compare_html(results, amount, term),
                        parse_mode=ParseMode.HTML, reply_markup=kb)


async def cmd_demo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _run(update, ctx, ctx.args, demo=True)


async def cmd_portfolio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _run(update, ctx, ctx.args, demo=False)


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _run(update, ctx, update.message.text.split(), demo=False)


async def on_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if not is_allowed(q.from_user.id):
        return
    _, risk = q.data.split("|")
    universe = ctx.chat_data.get("universe")
    if not universe:
        await q.message.reply_text("Данные устарели, пришли параметры заново.")
        return
    amount = ctx.chat_data["amount"]
    term = ctx.chat_data["term"]
    reinvest = ctx.chat_data.get("reinvest", DEFAULT_REINVEST)
    res = await asyncio.to_thread(build_portfolio, universe, amount, term, risk)
    await q.message.reply_text(render_portfolio_html(res, amount, term, reinvest, DEFAULT_TAX),
                               parse_mode=ParseMode.HTML)


def main():
    if not TG_TOKEN:
        raise SystemExit("Не задан TELEGRAM_TOKEN")
    if not ALLOWED:
        log.warning("ALLOWED_USERS пуст — бот открыт всем! Задай ALLOWED_USERS в переменных Railway.")
    app = Application.builder().token(TG_TOKEN).build()
    app.add_handler(CommandHandler(["start", "help"], cmd_start))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("demo", cmd_demo))
    app.add_handler(CommandHandler(["p", "portfolio"], cmd_portfolio))
    app.add_handler(CallbackQueryHandler(on_button, pattern=r"^show\|"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    log.info("Бот запущен (polling).")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
