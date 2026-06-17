#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram-бот «AI-сборщик портфеля облигаций» — интерактивная панель.
Тонкая обёртка над tinvest_bond_bot.py. Режим: long polling (Railway worker).

Переменные окружения (Railway → Variables):
  TELEGRAM_TOKEN  — токен бота от @BotFather (обязательно)
  TINVEST_TOKEN   — readonly-токен T-Invest (для реальных данных)
  ALLOWED_USERS   — список Telegram user id через запятую (задать!)
  MAX_CANDIDATES  — сколько облигаций тянуть из T-Invest (по умолчанию 60)
  INCLUDE_QUAL    — "1" чтобы включать бумаги для квалов

Команды:
  /start /help — открыть панель
  /id          — показать свой Telegram id
"""
import asyncio
import logging
import os
import time

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (Application, CommandHandler, MessageHandler,
                          CallbackQueryHandler, ContextTypes, filters)

from tinvest_bond_bot import (build_portfolio, demo_universe,
                              fetch_universe_from_tinvest,
                              render_portfolio_html,
                              default_panel, panel_text, panel_keyboard,
                              apply_panel_action)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

TG_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TINVEST_TOKEN = os.environ.get("TINVEST_TOKEN")
ALLOWED = {int(x) for x in os.environ.get("ALLOWED_USERS", "").replace(" ", "").split(",") if x}
MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "60"))
INCLUDE_QUAL = os.environ.get("INCLUDE_QUAL", "0") == "1"

_CACHE = {}
_CACHE_TTL = 300

HELP = (
    "<b>AI-сборщик портфеля облигаций</b>\n\n"
    "Настрой параметры кнопками и нажми «Собрать портфель».\n"
    "• Сумму, срок и реинвест меняй кнопками ➖/➕ или быстрыми пресетами\n"
    "• Риск и налог — переключателями\n"
    "• «📡 Данные» — демо или реальные котировки T-Invest\n\n"
    "Команды: /start — панель, /id — твой Telegram id\n\n"
    "⚠️ Это не инвестиционная рекомендация. Бот считает план, заявки не выставляет."
)


def is_allowed(uid):
    return (not ALLOWED) or (uid in ALLOWED)


def kb(state):
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(t, callback_data=d) for t, d in row] for row in panel_keyboard(state)])


def back_kb():
    return InlineKeyboardMarkup([[InlineKeyboardButton("◀ Изменить параметры", callback_data="back")]])


def get_universe(term, demo):
    if demo:
        return demo_universe()
    if not TINVEST_TOKEN:
        raise RuntimeError("не задан TINVEST_TOKEN")
    key = (round(term), INCLUDE_QUAL, MAX_CANDIDATES)
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < _CACHE_TTL:
        return hit[1]
    uni = fetch_universe_from_tinvest(TINVEST_TOKEN, term, MAX_CANDIDATES, INCLUDE_QUAL, 4, verbose=False)
    _CACHE[key] = (now, uni)
    return uni


async def cmd_start(update, ctx):
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("Доступ закрыт. Узнай свой id командой /id.")
        return
    state = ctx.chat_data.setdefault("panel", default_panel())
    await update.message.reply_text(panel_text(state), parse_mode=ParseMode.HTML, reply_markup=kb(state))


async def cmd_help(update, ctx):
    await update.message.reply_text(HELP, parse_mode=ParseMode.HTML)


async def cmd_id(update, ctx):
    u = update.effective_user
    await update.message.reply_text(
        f"Твой Telegram id: <code>{u.id}</code>\nВпиши его в ALLOWED_USERS на Railway.",
        parse_mode=ParseMode.HTML)


async def _safe_edit(q, text, markup, html=True):
    try:
        await q.edit_message_text(
            text, parse_mode=(ParseMode.HTML if html else None), reply_markup=markup)
    except BadRequest as e:
        if "not modified" in str(e).lower():
            return
        # текст не прошёл HTML-парсинг (например, тех. ошибка с угловыми скобками) —
        # повторяем без разметки, чтобы сообщение точно дошло
        if html:
            try:
                await q.edit_message_text(text, parse_mode=None, reply_markup=markup)
                return
            except BadRequest:
                pass
        raise


async def on_callback(update, ctx):
    q = update.callback_query
    if not is_allowed(q.from_user.id):
        await q.answer("Доступ закрыт", show_alert=True)
        return
    data = q.data
    state = ctx.chat_data.setdefault("panel", default_panel())

    if data == "noop":
        await q.answer()
        return

    if data == "back":
        await q.answer()
        await _safe_edit(q, panel_text(state), kb(state))
        return

    if data == "go":
        await q.answer()
        demo = state["src"] == "demo"
        src_label = "синтетике" if demo else "реальных данных T-Invest"
        await _safe_edit(q, f"⏳ Считаю на {src_label}…", None)
        try:
            universe = await asyncio.wait_for(
                asyncio.to_thread(get_universe, state["term"], demo), timeout=90)
            if not universe:
                raise RuntimeError("не нашёл подходящих бумаг под эти параметры")
            res = await asyncio.to_thread(build_portfolio, universe, state["amount"], state["term"], state["risk"])
        except asyncio.TimeoutError:
            await _safe_edit(
                q, "❌ T-Invest не ответил за 90 секунд. Скорее всего, у этого сервера "
                "нет доступа к торговому API T-Invest (хостинг за пределами РФ). "
                "Демо-режим работает; для реальных данных нужен сервер с доступом к РФ-инфраструктуре.",
                back_kb(), html=False)
            return
        except Exception as e:
            msg = str(e)
            if "No module" in msg or "t_tech" in msg or "tinkoff" in msg:
                msg = "библиотека T-Invest не установлена на сервере."
            elif "UNAVAILABLE" in msg.upper():
                msg = ("T-Invest вернул «UNAVAILABLE» — сервис недоступен с этого хостинга "
                       "(вероятно, нет доступа к РФ-инфраструктуре из дата-центра). "
                       "Демо-режим работает.")
            await _safe_edit(q, f"❌ Не получилось: {msg}", back_kb(), html=False)
            return
        text = render_portfolio_html(res, state["amount"], state["term"], state["reinvest"], state["tax"])
        await _safe_edit(q, text, back_kb())
        return

    changed = apply_panel_action(state, data)
    await q.answer()
    if changed:
        await _safe_edit(q, panel_text(state), kb(state))


def main():
    if not TG_TOKEN:
        raise SystemExit("Не задан TELEGRAM_TOKEN")
    if not ALLOWED:
        log.warning("ALLOWED_USERS пуст — бот открыт всем! Задай ALLOWED_USERS в Railway.")
    app = Application.builder().token(TG_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, cmd_start))
    log.info("Бот запущен (polling).")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
