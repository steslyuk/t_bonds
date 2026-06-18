#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-Invest bond portfolio bot (read-only).

Принимает те же параметры, что и калькулятор (сумма, срок, риск, ставка
реинвеста, налог), ходит в T-Invest API за реальными облигациями, считает
YTM/дюрацию из живых купонных потоков и стакана, собирает портфель тем же
оптимизатором (LP-релаксация -> округление -> добивка). Заявки НЕ выставляет.

Запуск:
  # 1) без токена, синтетика — проверить логику:
  python3 tinvest_bond_bot.py --demo --amount 1500000 --term 3 --risk mod

  # 2) реальные данные (нужен read-only токен в переменной окружения):
  export TINVEST_TOKEN="t.ВАШ_ТОКЕН"
  pip install tinkoff-investments
  python3 tinvest_bond_bot.py --amount 1500000 --term 3 --risk mod

Токен читается только из переменной окружения TINVEST_TOKEN и нигде не печатается.
Рекомендуется readonly-токен: вывести деньги через него нельзя, торговать тоже.
"""
import argparse
import datetime as dt
import json
import math
import os
import sys
import time

NOMINAL_DEFAULT = 1000.0

# ----------------------------- профили риска -----------------------------
PROFILES = {
    "cons": dict(label="Консервативный", minAvg=7.0, floor=6, minLiq=1, sectorCap=0.60, issuerCap=0.25),
    "mod":  dict(label="Умеренный",       minAvg=5.5, floor=4, minLiq=1, sectorCap=0.40, issuerCap=0.22),
    "agg":  dict(label="Агрессивный",     minAvg=4.0, floor=3, minLiq=1, sectorCap=0.45, issuerCap=0.25),
}
LETTER = {10: "AAA", 9: "AA", 8: "A", 7: "BBB", 6: "BB", 5: "BB-", 4: "B+", 3: "B", 2: "CCC", 1: "C", 0: "D"}
SPREAD = {5: 0.0008, 4: 0.0015, 3: 0.0030, 2: 0.0060, 1: 0.0120}
SLOPE  = {5: 0.004,  4: 0.008,  3: 0.020,  2: 0.050,  1: 0.100}
FILL0, BAND, RBUF = 0.30, 0.50, 0.06


def letter_from_score(n):
    return LETTER.get(max(0, min(10, round(n))), "—")


def rating_group(n):
    if n >= 9.5: return "AAA"
    if n >= 8.5: return "AA"
    if n >= 7.5: return "A"
    if n >= 6.5: return "BBB"
    if n >= 5.5: return "BB"
    if n >= 3.5: return "B"
    return "≤CCC"


# ----------------------------- математика облигаций -----------------------------
def ytm_from_cashflows(price_dirty, flows):
    """flows: list of (t_years, amount). Решаем для y: sum(cf/(1+y)^t)=price."""
    if price_dirty <= 0 or not flows:
        return None

    def npv(y):
        return sum(cf / (1.0 + y) ** t for t, cf in flows) - price_dirty

    lo, hi = -0.95, 3.0
    flo, fhi = npv(lo), npv(hi)
    if flo * fhi > 0:
        return None  # нет корня в диапазоне
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = npv(mid)
        if abs(fm) < 1e-7:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def modified_duration(price_dirty, flows, y):
    """Macaulay/(1+y) в годах."""
    if y is None or price_dirty <= 0:
        return None
    pv_t = sum(t * cf / (1.0 + y) ** t for t, cf in flows)
    pv = sum(cf / (1.0 + y) ** t for t, cf in flows)
    if pv <= 0:
        return None
    mac = pv_t / pv
    return mac / (1.0 + y)


# ----------------------------- LP: двухфазный симплекс -----------------------------
def simplex_max(c, constraints):
    """maximize c.x s.t. {a,type('<='|'>='|'='),b}, x>=0."""
    n = len(c)
    rows = []
    for con in constraints:
        a, b, t = list(con["a"]), con["b"], con["type"]
        if b < 0:
            a = [-v for v in a]; b = -b
            t = ">=" if t == "<=" else "<=" if t == ">=" else "="
        rows.append({"a": a, "b": b, "type": t})
    col = n
    meta = []
    for r in rows:
        if r["type"] == "<=":
            r["s"] = col; col += 1; meta.append("slack"); r["basis"] = r["s"]; r["art"] = None
        elif r["type"] == ">=":
            r["s"] = col; col += 1; meta.append("surplus")
            r["art"] = col; col += 1; meta.append("art"); r["basis"] = r["art"]
        else:
            r["s"] = None; r["art"] = col; col += 1; meta.append("art"); r["basis"] = r["art"]
    total = col
    T = []
    for r in rows:
        row = [0.0] * (total + 1)
        for j in range(n):
            row[j] = r["a"][j]
        if r.get("s") is not None:
            row[r["s"]] = 1.0 if r["type"] == "<=" else -1.0
        if r.get("art") is not None:
            row[r["art"]] = 1.0
        row[total] = r["b"]
        T.append(row)
    basis = [r["basis"] for r in rows]

    def pivot(prow, pcol):
        pv = T[prow][pcol]
        T[prow] = [v / pv for v in T[prow]]
        for i in range(len(T)):
            if i == prow:
                continue
            f = T[i][pcol]
            if abs(f) > 1e-12:
                T[i] = [T[i][j] - f * T[prow][j] for j in range(total + 1)]
        basis[prow] = pcol

    def optimize(cost):
        for _ in range(20000):
            red = [0.0] * total
            for j in range(total):
                cb = sum(cost[basis[i]] * T[i][j] for i in range(len(T)))
                red[j] = cb - cost[j]
            pcol = next((j for j in range(total) if red[j] > 1e-9), -1)
            if pcol < 0:
                return "optimal"
            prow, best = -1, float("inf")
            for i in range(len(T)):
                if T[i][pcol] > 1e-9:
                    ratio = T[i][total] / T[i][pcol]
                    if ratio < best - 1e-12 or (abs(ratio - best) < 1e-12 and (prow < 0 or basis[i] < basis[prow])):
                        best, prow = ratio, i
            if prow < 0:
                return "unbounded"
            pivot(prow, pcol)
        return "optimal"

    if any(m == "art" for m in meta):
        c1 = [0.0] * total
        for k, m in enumerate(meta):
            if m == "art":
                c1[n + k] = 1.0
        optimize(c1)
        w = 0.0
        for i in range(len(basis)):
            idx = basis[i] - n
            if 0 <= idx < len(meta) and meta[idx] == "art":
                w += T[i][total]
        if w > 1e-6:
            return {"status": "infeasible"}
    c2 = [0.0] * total
    for j in range(n):
        c2[j] = -c[j]
    for k, m in enumerate(meta):
        if m == "art":
            c2[n + k] = 1e9
    if optimize(c2) == "unbounded":
        return {"status": "unbounded"}
    x = [0.0] * n
    for i in range(len(basis)):
        if basis[i] < n:
            x[basis[i]] = T[i][total]
    obj = sum(c[j] * x[j] for j in range(n))
    return {"status": "optimal", "x": x, "obj": obj}


# ----------------------------- сборка портфеля -----------------------------
def build_portfolio(universe, amount, term_y, profile_key):
    cfg = PROFILES[profile_key]
    pool = [b for b in universe if b["internal"] >= cfg["floor"] and b["liq"] >= cfg["minLiq"]]
    lo, hi = max(0.3, term_y * 0.25), term_y * 2.2
    dp = [b for b in pool if lo <= b["durationY"] <= hi]
    pool = dp if len(dp) >= 6 else [b for b in pool if b["durationY"] <= term_y * 2.8]
    empty = dict(cfg=cfg, holdings=[], invested=0, cash=amount, ytm=0, duration=0,
                 rating=0, couponIncome=0, positions=0, sectors={}, ratings={}, ratingMet=False, gap=0)
    if not pool:
        return empty

    # fixed-impact цена/доходность для линейной модели
    P = []
    for b in pool:
        nominal = b.get("nominal", NOMINAL_DEFAULT)
        imp0 = SPREAD[b["liq"]] + SLOPE[b["liq"]] * FILL0
        price = nominal * (b["pricePct"] * (1 + imp0) + b["aciPct"]) / 100
        drag = min(1.2, (SPREAD[b["liq"]] + SLOPE[b["liq"]] * FILL0 / 2) / b["durationY"] * 100)
        P.append(dict(b, price=price, yEff=b["ytm"] - drag, nominal=nominal))

    n = len(P)
    Dlo, Dhi, Rmin = term_y * (1 - BAND), term_y * (1 + BAND), cfg["minAvg"]
    cons = []
    cons.append({"a": [b["price"] for b in P], "type": "<=", "b": amount})
    cons.append({"a": [b["price"] for b in P], "type": ">=", "b": 0.90 * amount})
    cons.append({"a": [b["price"] * (b["internal"] - Rmin) for b in P], "type": ">=", "b": 0})
    cons.append({"a": [b["price"] * (b["durationY"] - Dhi) for b in P], "type": "<=", "b": 0})
    cons.append({"a": [b["price"] * (b["durationY"] - Dlo) for b in P], "type": ">=", "b": 0})
    for s in {b["sector"] for b in P}:
        cons.append({"a": [b["price"] if b["sector"] == s else 0 for b in P], "type": "<=", "b": cfg["sectorCap"] * amount})
    for j in {b["issuer"] for b in P}:
        cons.append({"a": [b["price"] if b["issuer"] == j else 0 for b in P], "type": "<=", "b": cfg["issuerCap"] * amount})
    # Диверсификация: максимальный вес одной бумаги зависит от профиля
    # cons: 4% => минимум 25 бумаг
    # mod:  5% => минимум 20 бумаг
    # agg:  6% => минимум 17 бумаг (больше концентрации в доходных ВДО)
    # Лесенка потолков веса (умная горка): топ по доходности — крупнее, хвост — мельче.
    _rank = {idx: r for r, idx in enumerate(sorted(range(n), key=lambda k: -P[k]["yEff"]))}
    def _wcap(i):
        r = _rank[i]
        return 0.075 if r < 3 else 0.055 if r < 8 else 0.035

    c = [b["price"] * b["yEff"] for b in P]

    ub0 = [int(amount // b["price"]) for b in P]
    # передаём верхние границы как ограничения
    cons_lp = list(cons)
    for i in range(n):
        if ub0[i] < 1e8:
            a = [0] * n; a[i] = 1
            cons_lp.append({"a": a, "type": "<=", "b": ub0[i]})
    lp = None
    for _relax in (1.0, 1.4, 2.0, 100.0):
        cons_try = list(cons_lp)
        for i in range(n):
            cap = min(1.0, _wcap(i) * _relax)
            a = [0.0] * n; a[i] = P[i]["price"]
            cons_try.append({"a": a, "type": "<=", "b": cap * amount})
        lp = simplex_max(c, cons_try)
        if lp["status"] == "optimal":
            break
    if not lp or lp["status"] != "optimal":
        return empty
    investedLP = sum(P[i]["price"] * lp["x"][i] for i in range(n))
    LPyield = lp["obj"] / investedLP if investedLP else 0

    lots = [int(v + 1e-9) for v in lp["x"]]

    def cost(i):
        b = P[i]
        fill = min(1, (lots[i] + 1) / b["depthLots"])
        return b["nominal"] * (b["pricePct"] * (1 + SPREAD[b["liq"]] + SLOPE[b["liq"]] * fill) + b["aciPct"]) / 100

    invested = sum(lots[i] * P[i]["price"] for i in range(n))
    secCost, issCost = {}, {}
    for i, b in enumerate(P):
        secCost[b["sector"]] = secCost.get(b["sector"], 0) + lots[i] * b["price"]
        issCost[b["issuer"]] = issCost.get(b["issuer"], 0) + lots[i] * b["price"]

    def avgR():
        s = sum(P[i]["internal"] * lots[i] * P[i]["price"] for i in range(n))
        t = sum(lots[i] * P[i]["price"] for i in range(n))
        return s / t if t > 0 else 10

    def wDur():
        s = sum(P[i]["durationY"] * lots[i] * P[i]["price"] for i in range(n))
        t = sum(lots[i] * P[i]["price"] for i in range(n))
        return s / t if t > 0 else term_y

    # добивка бюджета
    for _ in range(60000):
        rem = amount - invested
        deficit = Rmin - avgR()
        wd = wDur()
        outside = abs(wd - term_y) > BAND * term_y
        dir_ = (term_y > wd) - (term_y < wd)
        best, bs = -1, -1e18
        for i in range(n):
            b = P[i]; cc = cost(i)
            if cc > rem: continue
            if secCost.get(b["sector"], 0) + cc > cfg["sectorCap"] * amount + 1e-6: continue
            if issCost.get(b["issuer"], 0) + cc > cfg["issuerCap"] * amount + 1e-6: continue
            cur_w = (lots[i] * P[i]["price"]) / amount
            if cur_w >= _wcap(i) * 1.1: continue  # добивка не превышает потолок лесенки * 1.4
            sc = b["ytm"] + (3 + 6 * deficit if deficit > 0 else 0) * (b["internal"] - Rmin)
            if outside:
                sc += max(-4, min(4, 0.9 * dir_ * (b["durationY"] - wd)))
            sc -= (issCost.get(b["issuer"], 0) / amount) * 4
            if sc > bs:
                bs, best = sc, i
        if best < 0:
            break
        cc = cost(best); lots[best] += 1; invested += cc
        secCost[P[best]["sector"]] += cc; issCost[P[best]["issuer"]] += cc

    # подтяжка рейтинга
    for _ in range(3000):
        if avgR() >= Rmin:
            break
        drop = -1
        for i in range(n):
            if lots[i] > 0 and (drop < 0 or P[i]["internal"] < P[drop]["internal"]):
                drop = i
        if drop < 0:
            break
        lots[drop] -= 1; invested -= P[drop]["price"]
        secCost[P[drop]["sector"]] -= P[drop]["price"]; issCost[P[drop]["issuer"]] -= P[drop]["price"]
        add = -1
        for i in range(n):
            b = P[i]
            if b["internal"] <= Rmin: continue
            cc = cost(i)
            if cc > amount - invested: continue
            if secCost.get(b["sector"], 0) + cc > cfg["sectorCap"] * amount + 1e-6: continue
            if issCost.get(b["issuer"], 0) + cc > cfg["issuerCap"] * amount + 1e-6: continue
            if add < 0 or b["ytm"] > P[add]["ytm"]:
                add = i
        if add >= 0:
            cc = cost(add); lots[add] += 1; invested += cc
            secCost[P[add]["sector"]] += cc; issCost[P[add]["issuer"]] += cc

    # финальная подрезка: отчётные цены (по факт. стакану) могут чуть превысить
    # бюджет относительно модельных — убираем самые мелкие лоты, пока влезаем
    def actual_total():
        tot = 0.0
        for i, b in enumerate(P):
            x = lots[i]
            if x <= 0:
                continue
            fill = min(1, x / b["depthLots"])
            tot += b["nominal"] * (b["pricePct"] * (1 + SPREAD[b["liq"]] + SLOPE[b["liq"]] * fill) + b["aciPct"]) / 100 * x
        return tot

    for _ in range(10000):
        if actual_total() <= amount:
            break
        worst = -1
        for i in range(n):
            if lots[i] > 0:
                ci = lots[i] * P[i]["price"]
                if worst < 0 or ci < lots[worst] * P[worst]["price"]:
                    worst = i
        if worst < 0:
            break
        lots[worst] -= 1

    holdings, sY, sR, sD, couponIncome, inv = [], 0, 0, 0, 0, 0
    for i, b in enumerate(P):
        x = lots[i]
        cc_check = b["nominal"] * (b["pricePct"] + b["aciPct"]) / 100 * x
        if x <= 0 or cc_check < 0.003 * amount:  # убираем позиции с весом < 0.3%
            continue
        fill = min(1, x / b["depthLots"])
        ccost = b["nominal"] * (b["pricePct"] * (1 + SPREAD[b["liq"]] + SLOPE[b["liq"]] * fill) + b["aciPct"]) / 100 * x
        ytmEff = b["ytm"] - min(1.2, (SPREAD[b["liq"]] + SLOPE[b["liq"]] * fill / 2) / b["durationY"] * 100)
        inv += ccost; sY += ytmEff * ccost; sR += b["internal"] * ccost; sD += b["durationY"] * ccost
        couponIncome += x * b["nominal"] * b["couponPct"] / 100
        holdings.append(dict(id=b["id"], name=b["name"], issuer=b["issuer"], sector=b["sector"],
                             internal=b["internal"], letter=letter_from_score(b["internal"]),
                             lots=x, cost=ccost, durationY=b["durationY"], ytmEff=ytmEff,
                             couponRate=b["couponPct"], weight=0.0))
    for h in holdings:
        h["weight"] = h["cost"] / inv if inv else 0
    holdings.sort(key=lambda h: -h["cost"])
    sectors, ratings = {}, {}
    for h in holdings:
        sectors[h["sector"]] = sectors.get(h["sector"], 0) + h["weight"]
        g = rating_group(h["internal"]); ratings[g] = ratings.get(g, 0) + h["weight"]
    rating = sR / inv if inv else 0
    intFixed = sum(P[i]["price"] * P[i]["yEff"] * lots[i] for i in range(n))
    intInv = sum(P[i]["price"] * lots[i] for i in range(n))
    gap = max(0, LPyield - (intFixed / intInv if intInv else 0))
    return dict(cfg=cfg, holdings=holdings, invested=inv, cash=amount - inv,
                ytm=(sY / inv if inv else 0), duration=(sD / inv if inv else 0),
                rating=rating, couponIncome=couponIncome, positions=len(holdings),
                sectors=sectors, ratings=ratings, ratingMet=(rating + 0.03 >= Rmin), gap=gap)


def project_value(holdings, amount, term_y, reinvest_rate, tax_on):
    months = max(1, round(term_y * 12))
    rM = (1 + reinvest_rate / 100) ** (1 / 12) - 1
    tax = (0.15 if amount > 5_000_000 else 0.13) if tax_on else 0.0
    monthly = sum(h["lots"] * h.get("nominal", NOMINAL_DEFAULT) * h["couponRate"] / 100 / 12 for h in holdings)
    cash, coupon_gross, coupon_tax = 0.0, 0.0, 0.0
    for _ in range(months):
        cash *= (1 + rM)
        coupon_gross += monthly
        coupon_tax += monthly * tax
        cash += monthly * (1 - tax)
    redeem, gain_tax = 0.0, 0.0
    for h in holdings:
        par = h["lots"] * h.get("nominal", NOMINAL_DEFAULT)
        redeem += par
        if par > h["cost"]:
            gain_tax += (par - h["cost"]) * tax
    fv = redeem - gain_tax + cash
    invested = sum(h["cost"] for h in holdings)
    return dict(fv=fv, couponGross=coupon_gross, totalTax=coupon_tax + gain_tax, taxRate=tax,
                annualizedNet=(fv / invested) ** (1 / term_y) - 1 if invested else 0)


# ----------------------------- демо-вселенная -----------------------------
def demo_universe():
    raw = [
        ("SU26240", "ОФЗ 26240", "Минфин РФ", "Государство", 10, 12.25, 84.1, 1.1, 15.4, 5.8, 9000, 5),
        ("SU26226", "ОФЗ 26226", "Минфин РФ", "Государство", 10, 7.95, 92.0, 0.7, 15.7, 2.6, 9000, 5),
        ("SU26219", "ОФЗ 26219", "Минфин РФ", "Государство", 10, 7.75, 96.4, 0.5, 15.9, 1.1, 9000, 5),
        ("SBER3", "Сбербанк 001P", "Сбербанк", "Банки", 9, 16.5, 100.6, 0.9, 16.8, 2.2, 4000, 5),
        ("GAZPR8", "Газпром капитал", "Газпром", "Нефтегаз", 9, 15.9, 99.1, 0.6, 17.2, 3.3, 2500, 4),
        ("RZD28", "РЖД 001P", "РЖД", "Транспорт", 9, 16.2, 99.8, 0.8, 17.0, 2.8, 2200, 4),
        ("LUK4", "ЛУКОЙЛ БО", "ЛУКОЙЛ", "Нефтегаз", 9, 15.7, 98.9, 0.4, 17.4, 3.9, 1800, 4),
        ("MTS21", "МТС 001P", "МТС", "Телеком", 8, 17.4, 100.2, 1.0, 18.6, 2.1, 1600, 4),
        ("MAGN3", "Магнит БО", "Магнит", "Ритейл", 8, 17.0, 99.3, 0.5, 18.9, 1.7, 1400, 4),
        ("GMK7", "Норникель БО", "Норникель", "Металлургия", 8, 16.8, 98.6, 0.7, 19.2, 3.0, 1300, 4),
        ("SIBUR2", "СИБУР", "СИБУР", "Химия", 8, 17.2, 99.0, 0.6, 19.0, 2.5, 1100, 3),
        ("PHOS1", "ФосАгро БО", "ФосАгро", "Химия", 8, 16.9, 98.2, 0.9, 19.3, 3.4, 900, 3),
        ("AFK5", "АФК Система", "АФК Система", "Холдинги", 7, 19.5, 99.4, 1.1, 21.4, 1.9, 1500, 4),
        ("PIK4", "ПИК БО", "ПИК", "Девелопмент", 7, 19.8, 98.0, 0.8, 22.1, 1.5, 1200, 3),
        ("ROSS2", "Россети БО", "Россети", "Энергетика", 7, 18.9, 99.7, 0.5, 20.6, 3.1, 1000, 3),
        ("BLIZ1", "Балт. лизинг", "Балтийский лизинг", "Лизинг", 7, 20.2, 98.8, 1.0, 22.3, 2.0, 700, 3),
        ("VUSH2", "ВУШ", "ВУШ", "Транспорт", 7, 20.0, 99.1, 0.7, 21.9, 1.3, 800, 3),
        ("EUPL1", "Европлан БО", "Европлан", "Лизинг", 6, 21.5, 98.5, 0.9, 24.1, 1.8, 600, 3),
        ("GTLK9", "ГТЛК БО", "ГТЛК", "Лизинг", 6, 21.0, 97.4, 1.0, 24.6, 2.7, 700, 3),
        ("SAMO2", "Самолёт БО", "Самолёт", "Девелопмент", 6, 22.5, 96.8, 0.8, 25.7, 1.6, 650, 3),
        ("ETRN3", "ЕвроТранс", "ЕвроТранс", "Нефтегаз", 6, 22.0, 97.9, 0.6, 24.4, 2.2, 500, 2),
        ("LSR4", "ЛСР БО", "ЛСР", "Девелопмент", 6, 21.8, 97.0, 1.1, 25.2, 2.4, 480, 2),
        ("DELI2", "Делимобиль", "Делимобиль", "Транспорт", 4, 24.0, 96.2, 0.9, 28.4, 1.4, 350, 2),
        ("BRUS1", "Брусника", "Брусника", "Девелопмент", 4, 25.0, 95.5, 1.0, 29.6, 1.2, 300, 2),
        ("AERO1", "Аэрофьюэлз", "Аэрофьюэлз", "Нефтегаз", 4, 24.5, 96.0, 0.7, 28.9, 1.7, 250, 2),
        ("MVID3", "М.Видео БО", "М.Видео", "Ритейл", 3, 26.5, 93.8, 1.2, 32.1, 1.1, 280, 2),
        ("VIS3", "ВИС Финанс", "Группа ВИС", "Инфраструктура", 4, 24.8, 95.9, 0.8, 29.1, 1.9, 220, 1),
        ("SEGE2", "Сегежа БО", "Сегежа", "Лесопром", 3, 27.0, 92.0, 1.0, 33.8, 1.0, 200, 1),
        ("GTLK10", "ГТЛК БО-П", "ГТЛК", "Лизинг", 6, 21.3, 96.5, 0.8, 25.4, 3.9, 600, 3),
        ("OKEY1", "О'КЕЙ БО", "О'КЕЙ", "Ритейл", 5, 23.5, 97.5, 0.5, 26.2, 3.3, 300, 2),
        ("AZBU1", "Азбука Вкуса", "Азбука Вкуса", "Ритейл", 5, 24.0, 97.0, 0.6, 26.8, 2.9, 320, 2),
        ("TECHL1", "Техно Лизинг", "Техно Лизинг", "Лизинг", 4, 27.5, 95.0, 0.9, 30.6, 2.7, 240, 2),
        ("BRUS2", "Брусника БО-П", "Брусника", "Девелопмент", 4, 26.8, 95.2, 0.8, 30.1, 2.6, 260, 2),
    ]
    keys = ["id", "name", "issuer", "sector", "internal", "couponPct", "pricePct", "aciPct", "ytm", "durationY", "depthLots", "liq"]
    out = []
    for r in raw:
        d = dict(zip(keys, r))
        d["nominal"] = NOMINAL_DEFAULT
        out.append(d)
    return out


# ----------------------------- реальные данные T-Invest -----------------------------
# ВНИМАНИЕ: этот блок не тестировался против живого API (нет сети в окружении сборки).
# Запускай сначала --demo, затем реальный режим и сверяй вывод.
RATING_MAP = {
    # заполни вручную известными рейтингами по ISIN или тикеру: "RU000A..." : 7
    # это временная заглушка вместо твоего PD-скора / фида агентств
}


def _to_dec(q):
    """Quotation/MoneyValue -> float (units + nano/1e9)."""
    if q is None:
        return 0.0
    units = getattr(q, "units", 0) or 0
    nano = getattr(q, "nano", 0) or 0
    return units + nano / 1e9


def fetch_universe_from_tinvest(token, term_y, max_candidates, include_qual, unknown_rating, verbose=True):
    # SSL решается через GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=/root/tbank_chain.pem в /root/tbonds.env
    try:
        from t_tech.invest import Client, InstrumentStatus
    except ImportError:
        from tinkoff.invest import Client, InstrumentStatus
    import warnings
    now = dt.datetime.now(dt.timezone.utc)
    horizon = now + dt.timedelta(days=int(term_y * 365 * 2.8) + 30)
    universe = []

    with Client(token) as client:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all_bonds = client.instruments.bonds(
                instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE).instruments

        cand = []
        for b in all_bonds:
            try:
                if b.currency != "rub": continue
                if not b.api_trade_available_flag: continue
                if b.for_qual_investor_flag and not include_qual: continue
                mat = getattr(b, "maturity_date", None)
                if mat is None or mat <= now or mat > horizon: continue
                cand.append(b)
            except Exception:
                continue
        # Балансируем выборку: все ОФЗ + равномерно по уровням риска
        ofz  = [b for b in cand if "ОФЗ" in (b.name or "") or (getattr(b,"ticker","") or "").startswith("SU")]
        rl1  = [b for b in cand if getattr(b,"risk_level",2)==1 and b not in set(ofz)]
        rl2  = [b for b in cand if getattr(b,"risk_level",2)==2]
        rl3  = [b for b in cand if getattr(b,"risk_level",2)==3]
        # берём все ОФЗ + по трети остатка из каждого уровня риска
        per_bucket = max(10, (max_candidates - len(ofz)) // 3)
        cand = ofz + rl1[:per_bucket] + rl2[:per_bucket] + rl3[:per_bucket]
        if verbose:
            print(f"[i] кандидатов: {len(cand)} (ОФЗ={len(ofz)} rl1={min(len(rl1),per_bucket)} rl2={min(len(rl2),per_bucket)} rl3={min(len(rl3),per_bucket)})", file=sys.stderr)

        figis = [b.figi for b in cand]
        last = {}
        for i in range(0, len(figis), 100):
            try:
                for lp in client.market_data.get_last_prices(figi=figis[i:i+100]).last_prices:
                    v = _to_dec(lp.price)
                    if v > 0: last[lp.figi] = v
            except Exception:
                pass
            time.sleep(0.1)

        for b in cand:
            try:
                price_pct = last.get(b.figi)
                if not price_pct or price_pct <= 0: continue
                nominal = _to_dec(b.nominal) or NOMINAL_DEFAULT
                aci = _to_dec(getattr(b, "aci_value", None))
                aci_pct = (aci / nominal * 100) if nominal else 0
                dirty = nominal * price_pct / 100 + aci
                mat = b.maturity_date
                cps = []
                for _attempt in range(3):
                    try:
                        cps = client.instruments.get_bond_coupons(figi=b.figi, from_=now, to=mat).events
                        break
                    except Exception as ex:
                        if "RESOURCE_EXHAUSTED" in str(ex):
                            time.sleep(2)
                        else:
                            break
                flows, annual_coupon_rub = [], 0.0
                # флоатеры: будущие купоны = 0, заполняем последним известным
                last_coupon = 0.0
                for cp in cps:
                    a = _to_dec(cp.pay_one_bond)
                    if a > 0:
                        last_coupon = a
                for cp in cps:
                    amt = _to_dec(cp.pay_one_bond)
                    if amt <= 0:
                        amt = last_coupon  # подставляем последний известный купон
                    if amt <= 0: continue
                    t = (cp.coupon_date - now).days / 365.0
                    if t > 0:
                        flows.append((t, amt))
                        if t <= 1.0: annual_coupon_rub += amt
                t_mat = (mat - now).days / 365.0
                if t_mat <= 0: continue
                flows.append((t_mat, nominal))
                if len(flows) < 2: continue
                time.sleep(0.15)
                y = ytm_from_cashflows(dirty, flows)
                dur = modified_duration(dirty, flows, y)
                if y is None or dur is None or dur <= 0 or y < 0: continue
                ytm_pct = y * 100
                if ytm_pct < 3 or ytm_pct > 60: continue
                liq, depth = 1, 50
                try:
                    ob = client.market_data.get_order_book(figi=b.figi, depth=20)
                    asks, bids = ob.asks, ob.bids
                    depth = max(1, sum(a.quantity for a in asks)) if asks else 1
                    if asks and bids:
                        spr = (_to_dec(asks[0].price) - _to_dec(bids[0].price)) / ((_to_dec(asks[0].price) + _to_dec(bids[0].price)) / 2 or 1)
                        liq = 5 if spr < 0.001 else 4 if spr < 0.003 else 3 if spr < 0.007 else 2 if spr < 0.02 else 1
                    time.sleep(0.03)
                except Exception:
                    pass
                name = b.name or getattr(b, "isin", "") or ""
                ticker = getattr(b, "ticker", "") or ""
                is_ofz = "ОФЗ" in name or ticker.startswith("SU")
                rmap = RATING_MAP.get(getattr(b, "isin", ""), RATING_MAP.get(ticker))
                if is_ofz:
                    internal = 10
                elif rmap:
                    internal = rmap
                else:
                    rl = getattr(b, "risk_level", 2)
                    internal = {0: 10, 1: 8, 2: 6, 3: 4}.get(int(rl) if rl is not None else 2, 6)
                if annual_coupon_rub <= 0 and len(flows) > 1:
                    annual_coupon_rub = sum(amt for t, amt in flows[:-1]) / t_mat
                universe.append(dict(
                    id=getattr(b, "isin", ticker) or ticker, name=name[:40],
                    issuer=" ".join(name.split()[:2]).upper(),
                    sector=(getattr(b, "sector", "") or "прочее"),
                    internal=internal, couponPct=(annual_coupon_rub / nominal * 100) if nominal else 0,
                    pricePct=price_pct, aciPct=aci_pct, ytm=ytm_pct, durationY=dur,
                    depthLots=depth, liq=liq, nominal=nominal))
            except Exception as e:
                if verbose:
                    print(f"[skip] {getattr(b,'isin','?')}: {e}", file=sys.stderr)
                continue
    if verbose:
        print(f"[i] готово, бумаг с полными данными: {len(universe)}", file=sys.stderr)
    return universe

# ----------------------------- вывод -----------------------------
def fmt_rub(v):
    return f"{round(v):,}".replace(",", " ") + " ₽"


def print_report(res, amount, term, risk, reinvest, tax_on):
    proj = project_value(res["holdings"], amount, term, reinvest, tax_on)
    cfg = res["cfg"]
    print("=" * 78)
    print(f"  ПОРТФЕЛЬ — {cfg['label']} | сумма {fmt_rub(amount)} | срок {term} лет")
    print("=" * 78)
    print(f"  Эффективная доходность : {res['ytm']:.2f} %   (разрыв с LP-оптимумом {res['gap']:.3f} п.п.)")
    print(f"  Средний рейтинг        : {letter_from_score(res['rating'])} ({res['rating']:.2f})"
          f"  {'✓ цель достигнута' if res['ratingMet'] else '✗ ниже целевого'}")
    print(f"  Дюрация                : {res['duration']:.2f} лет (цель {term})")
    print(f"  Купонный доход / год    : {fmt_rub(res['couponIncome'])}")
    print(f"  Инвестировано / остаток : {fmt_rub(res['invested'])} / {fmt_rub(res['cash'])}")
    print(f"  Позиций                : {res['positions']}")
    print("-" * 78)
    print(f"  Купоны за срок (до нал.): {fmt_rub(proj['couponGross'])}")
    if tax_on:
        print(f"  Налог ({round(proj['taxRate']*100)}%)            : -{fmt_rub(proj['totalTax'])}")
    print(f"  ПРОГНОЗ ЧЕРЕЗ {term} ЛЕТ     : {fmt_rub(proj['fv'])}  (~{proj['annualizedNet']*100:.1f}% годовых нетто, реинвест {reinvest}%)")
    print("=" * 78)
    print(f"  {'Бумага':<22}{'Рейт':<6}{'Лот':>5}{'Дюр':>6}{'YTMэф':>8}{'Сумма':>13}{'Вес':>7}")
    print("-" * 78)
    for h in res["holdings"]:
        print(f"  {h['name'][:21]:<22}{h['letter']:<6}{h['lots']:>5}{h['durationY']:>6.1f}"
              f"{h['ytmEff']:>7.1f}%{fmt_rub(h['cost']):>13}{h['weight']*100:>6.1f}%")
    print("=" * 78)


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_portfolio_html(res, amount, term, reinvest, tax_on):
    """Полный портфель одного профиля для Telegram (parse_mode=HTML)."""
    proj = project_value(res["holdings"], amount, term, reinvest, tax_on)
    cfg = res["cfg"]
    if not res["holdings"]:
        return (f"<b>{_esc(cfg['label'])}</b>\nНе удалось собрать портфель под эти параметры "
                f"(не хватает подходящих бумаг). Попробуй другой срок/риск.")
    lines = []
    lines.append(f"<b>📊 {_esc(cfg['label'])}</b> · {_esc(fmt_rub(amount))} · {term:g} лет")
    flag = "✅ цель по рейтингу достигнута" if res["ratingMet"] else "⚠️ рейтинг ниже целевого"
    lines.append(
        f"Доходность: <b>{res['ytm']:.2f}%</b> годовых  (оптимум, разрыв {res['gap']:.3f} п.п.)\n"
        f"Рейтинг: <b>{letter_from_score(res['rating'])}</b> ({res['rating']:.2f}) — {flag}\n"
        f"Дюрация: {res['duration']:.2f} лет · купон/год: {_esc(fmt_rub(res['couponIncome']))}\n"
        f"Вложено: {_esc(fmt_rub(res['invested']))} · позиций: {res['positions']}"
    )
    tax_line = f"\nналог {round(proj['taxRate']*100)}%: −{fmt_rub(proj['totalTax'])}" if tax_on else ""
    lines.append(
        f"💰 Прогноз через {term:g} лет: <b>{_esc(fmt_rub(proj['fv']))}</b>\n"
        f"≈ {proj['annualizedNet']*100:.1f}% годовых нетто (реинвест {reinvest:g}%)"
        f"{_esc(tax_line)}"
    )
    rows = [f"{'Бумага':<16}{'Рейт':<5}{'Лот':>4}{'YTM':>7}{'Вес':>6}"]
    for h in res["holdings"]:
        rows.append(f"{_esc(h['name'][:15]):<16}{h['letter']:<5}{h['lots']:>4}"
                    f"{h['ytmEff']:>6.1f}%{h['weight']*100:>5.0f}%")
    lines.append("<pre>" + "\n".join(rows) + "</pre>")
    return "\n\n".join(lines)


def render_compare_html(results, amount, term):
    """Компактное сравнение трёх профилей."""
    lines = [f"<b>📊 Портфели · {_esc(fmt_rub(amount))} · {term:g} лет</b>"]
    rows = [f"{'Профиль':<14}{'Дох-сть':>8}{'Рейт':>6}{'Дюр':>6}"]
    names = {"cons": "Консерват.", "mod": "Умеренный", "agg": "Агрессивн."}
    for key in ("cons", "mod", "agg"):
        r = results.get(key)
        if not r or not r["holdings"]:
            rows.append(f"{names[key]:<14}{'—':>8}{'—':>6}{'—':>6}")
            continue
        rows.append(f"{names[key]:<14}{r['ytm']:>7.1f}%{letter_from_score(r['rating']):>6}{r['duration']:>5.1f}л")
    lines.append("<pre>" + "\n".join(rows) + "</pre>")
    lines.append("Нажми профиль ниже, чтобы увидеть полный состав 👇")
    return "\n".join(lines)


# ----------------------------- интерактивная панель (степперы + пресеты) -----------------------------
RISK_LABEL = {"cons": "Консервативный", "mod": "Умеренный", "agg": "Агрессивный"}


def default_panel():
    return {"amount": 1_500_000, "term": 3.0, "risk": "mod",
            "reinvest": 18.0, "tax": True, "src": "demo"}


def _amount_step(v):
    if v < 1_000_000: return 100_000
    if v < 3_000_000: return 250_000
    if v < 5_000_000: return 500_000
    if v < 20_000_000: return 1_000_000
    return 5_000_000


def step_amount(v, d):
    step = _amount_step(v if d > 0 else v - 1)
    nv = round((v + d * step) / step) * step
    return int(max(100_000, min(100_000_000, nv)))


def step_term(v, d):
    return max(0.5, min(15.0, round((v + d * 0.5) * 2) / 2))


def step_reinvest(v, d):
    return max(0.0, min(30.0, round(v + d)))


def apply_panel_action(s, data):
    """Меняет состояние панели по callback_data. Возвращает True, если что-то изменилось."""
    before = dict(s)
    if data == "a-": s["amount"] = step_amount(s["amount"], -1)
    elif data == "a+": s["amount"] = step_amount(s["amount"], +1)
    elif data.startswith("ap:"): s["amount"] = int(data[3:])
    elif data == "t-": s["term"] = step_term(s["term"], -1)
    elif data == "t+": s["term"] = step_term(s["term"], +1)
    elif data.startswith("tp:"): s["term"] = float(data[3:])
    elif data.startswith("r:"): s["risk"] = data[2:]
    elif data == "ri-": s["reinvest"] = step_reinvest(s["reinvest"], -1)
    elif data == "ri+": s["reinvest"] = step_reinvest(s["reinvest"], +1)
    elif data == "tax": s["tax"] = not s["tax"]
    elif data == "src": s["src"] = "real" if s["src"] == "demo" else "demo"
    return s != before


def panel_text(s):
    return (
        "<b>⚙️ Параметры портфеля</b>\n\n"
        f"💰 Сумма: <b>{_esc(fmt_rub(s['amount']))}</b>\n"
        f"📅 Срок: <b>{s['term']:g} лет</b>\n"
        f"⚖️ Риск: <b>{RISK_LABEL[s['risk']]}</b>\n"
        f"🔁 Реинвест купонов: <b>{s['reinvest']:g}%</b>\n"
        f"🧾 Налог: <b>{'учитывать' if s['tax'] else 'без налога'}</b>\n"
        f"📡 Данные: <b>{'демо' if s['src'] == 'demo' else 'реальные (T-Invest)'}</b>\n\n"
        "Настрой кнопками ниже и жми «Собрать портфель»."
    )


def panel_keyboard(s):
    """Возвращает раскладку как список рядов из (текст, callback_data)."""
    pick = lambda k: "✅ " if s["risk"] == k else ""
    return [
        [("➖", "a-"), (fmt_rub(s["amount"]), "noop"), ("➕", "a+")],
        [("300к", "ap:300000"), ("1 млн", "ap:1000000"), ("3 млн", "ap:3000000"), ("10 млн", "ap:10000000")],
        [("➖", "t-"), (f"срок: {s['term']:g} лет", "noop"), ("➕", "t+")],
        [("1", "tp:1"), ("2", "tp:2"), ("3", "tp:3"), ("5", "tp:5"), ("7", "tp:7")],
        [(pick("cons") + "Консерв.", "r:cons"), (pick("mod") + "Умерен.", "r:mod"), (pick("agg") + "Агресс.", "r:agg")],
        [("➖", "ri-"), (f"реинвест: {s['reinvest']:g}%", "noop"), ("➕", "ri+")],
        [(f"🧾 Налог: {'вкл ✅' if s['tax'] else 'выкл'}", "tax"),
         (f"📡 {'Демо' if s['src'] == 'demo' else 'Реальные'}", "src")],
        [("📊 Собрать портфель", "go")],
    ]


def main():
    ap = argparse.ArgumentParser(description="T-Invest read-only bond portfolio bot")
    ap.add_argument("--amount", type=float, default=1_500_000)
    ap.add_argument("--term", type=float, default=3, help="срок / целевая дюрация, лет")
    ap.add_argument("--risk", choices=["cons", "mod", "agg"], default="mod")
    ap.add_argument("--reinvest", type=float, default=18, help="ставка реинвеста купонов, %%")
    ap.add_argument("--tax", dest="tax", action="store_true", default=True)
    ap.add_argument("--no-tax", dest="tax", action="store_false")
    ap.add_argument("--demo", action="store_true", help="синтетические данные, без токена")
    ap.add_argument("--all-profiles", action="store_true", help="показать все три профиля")
    ap.add_argument("--max-candidates", type=int, default=60)
    ap.add_argument("--include-qual", action="store_true", help="включать бумаги для квалов")
    ap.add_argument("--unknown-rating", type=int, default=6, help="рейтинг для бумаг без записи в RATING_MAP")
    ap.add_argument("--json", action="store_true", help="вывод в JSON")
    args = ap.parse_args()

    if args.demo:
        universe = demo_universe()
    else:
        token = os.environ.get("TINVEST_TOKEN")
        if not token:
            print("Ошибка: задай токен в переменной окружения TINVEST_TOKEN "
                  "(или запусти с --demo).", file=sys.stderr)
            sys.exit(1)
        try:
            universe = fetch_universe_from_tinvest(
                token, args.term, args.max_candidates, args.include_qual, args.unknown_rating)
        except ImportError:
            print("Нужен пакет: pip install tinkoff-investments", file=sys.stderr)
            sys.exit(1)
        if not universe:
            print("Не удалось собрать вселенную бумаг (проверь фильтры/доступ).", file=sys.stderr)
            sys.exit(1)

    profiles = ["cons", "mod", "agg"] if args.all_profiles else [args.risk]
    results = {}
    for p in profiles:
        res = build_portfolio(universe, args.amount, args.term, p)
        results[p] = res
        if args.json:
            continue
        print_report(res, args.amount, args.term, p, args.reinvest, args.tax)

    if args.json:
        def clean(res):
            return {k: v for k, v in res.items() if k != "cfg"} | {"profile": res["cfg"]["label"]}
        print(json.dumps({p: clean(r) for p, r in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
