# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

"""
Kaggriculture agent v1 -- greedy task assignment.

Core idea: every turn we enumerate every useful thing that could be done on the
farm, give each one a coin value, then hand each unit (farmer + hired hands) the
task with the best value-per-turn, where a task d steps away costs d+1 turns.

Strategy thesis:
  * EGGS are the only product whose price does not collapse (log glut curve),
    so geese are the long-run engine.
  * MELON has the best coins-per-action of any crop but the market only absorbs
    ~158 units before hitting the $1 floor, so it is an early land grab.
  * FERTILIZER is a free byproduct of animals and sells high early.
  * Farm hands are almost free (fib cost: 10 hands = $143/day for 240 actions),
    so we hire aggressively. Most agents will not.
"""

import math

# --------------------------------------------------------------------------
# Game constants, transcribed from kaggriculture.py (NOT from the README --
# the source is the only ground truth).
# --------------------------------------------------------------------------
CROPS = {
    "WHEAT":      {"seed": 10,  "first": 2,  "maxday": 4,  "interval": 0, "maxyield": 6, "ongoing": False},
    "CARROT":     {"seed": 20,  "first": 2,  "maxday": 3,  "interval": 0, "maxyield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first": 8,  "maxday": 11, "interval": 1, "maxyield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first": 10, "maxday": 16, "interval": 2, "maxyield": 4, "ongoing": True},
    "MELON":      {"seed": 80,  "first": 10, "maxday": 10, "interval": 0, "maxyield": 6, "ongoing": False},
}

ANIMALS = {
    "GOOSE": {"cost": 300, "product": "EGG",  "structure": "COOP",    "first": 4, "interval": 1, "maxheld": 4},
    "COW":   {"cost": 400, "product": "MILK", "structure": "PASTURE", "first": 8, "interval": 2, "maxheld": 6},
    "SHEEP": {"cost": 500, "product": "WOOL", "structure": "PASTURE", "first": 6, "interval": 3, "maxheld": 6},
}

LAND_PRICES = [1000, 2000, 4000]
QUAD_BOUNDS = {  # quadrant -> (x0, y0) of its 5x5 block on a 10x10 board
    "NW": (0, 0), "NE": (5, 0), "SW": (0, 5), "SE": (5, 5),
}

# --------------------------------------------------------------------------
# Tunable policy knobs. These are the dials we turn on later days.
# --------------------------------------------------------------------------
CASH_RESERVE      = 400    # never spend below this, so we can always buy wheat
MAX_HANDS         = 11   # fib sum ~= $232/day, trivial against egg revenue      # cumulative fib cost of 9 hands is only $88/day
WHEAT_BUFFER      = 12     # keep this much wheat in the shed for feeding
MELON_LAST_DAY    = 19     # melon needs 10 days; planting later never harvests
GOOSE_LAST_DAY    = 24     # a goose bought later than this cannot pay back $300
SHED_SOFT_CAP     = 85     # start force-selling below the hard cap of 100
DROP_INV_AT       = 5      # route a unit to the shed once it carries this much


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def dist(a, b):
    """Manhattan distance. Movement is orthogonal, one tile per turn, and
    locked tiles are passable, so there is never a need for real pathfinding."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(src, dst):
    """Return the movement op that closes the larger axis gap first."""
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return ["EAST"] if dx > 0 else ["WEST"]
    if dy != 0:
        return ["SOUTH"] if dy > 0 else ["NORTH"]
    if dx != 0:
        return ["EAST"] if dx > 0 else ["WEST"]
    return ["PASS"]


def shed_tiles(board):
    """The four centre tiles that count as shed-adjacent. Three of them start
    LOCKED, but the shed is reachable from all of them anyway."""
    h = board // 2
    return [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]


def unlocked_tiles(farm, board):
    """Every (x, y) in a quadrant we actually own."""
    out = []
    for q in farm["unlocked_quadrants"]:
        x0, y0 = QUAD_BOUNDS[q]
        for y in range(y0, y0 + board // 2):
            for x in range(x0, x0 + board // 2):
                out.append((x, y))
    return out


SUPPLIES = ("WHEAT", "GOOSE", "COW", "SHEEP")


def inv_count(inv):
    """Sellable produce a unit is carrying.

    Carried WHEAT and livestock are SUPPLIES, not cargo -- counting them here
    made units fetch 10 wheat, trip the drop threshold, and put it straight back
    in the shed, looping forever while the animals starved."""
    return sum(v for k, v in inv.items() if k not in SUPPLIES)


# --------------------------------------------------------------------------
# Task generation -- the heart of the agent
# --------------------------------------------------------------------------
def build_tasks(obs, farm, priv, board, prices):
    """Enumerate every worthwhile action available on the farm this turn.

    Each task is a dict:
        pos    -- tile the unit must be standing on
        op     -- the action list to emit once there
        value  -- expected coins gained (used for priority)
        needs  -- item that must already be in the unit's inventory, or None
        tag    -- coarse category, used to avoid double-assigning a tile
    """
    day = obs["day"]
    tasks = []
    owned = unlocked_tiles(farm, board)
    shed = priv["shed"]

    n_geese = 0
    n_animals = 0
    n_plants = 0
    n_wheat_tiles = 0
    empty_structs = []
    empty_tiles = []

    for (x, y) in owned:
        tile = farm["tiles"][y][x]

        # ---- empty ground -------------------------------------------------
        if tile is None:
            empty_tiles.append((x, y))
            continue

        if not isinstance(tile, dict):
            continue
        kind = tile.get("kind")

        # ---- weeds: cheap to clear, frees a tile ---------------------------
        if kind == "WEED":
            tasks.append({"pos": (x, y), "op": ["DIG"], "value": 25,
                          "needs": None, "tag": ("tile", x, y)})
            continue

        # ---- crops ---------------------------------------------------------
        if kind == "PLANT":
            n_plants += 1
            if tile["crop"] == "WHEAT":
                n_wheat_tiles += 1
            cd = CROPS[tile["crop"]]
            age = day - tile["planted_day"]
            price = prices.get(tile["crop"], 1)

            if not tile["watered_today"]:
                # A plant already at 1 missed day dies tonight. That is a total
                # loss of the tile plus everything invested in it, so it
                # outranks literally every other task on the board.
                if tile["consecutive_unwatered"] >= 1:
                    val = 100000
                else:
                    # Inside the bonus window each watering is worth real units.
                    win_start = (cd["maxday"] + 1) // 2
                    if not cd["ongoing"] and win_start <= age <= cd["maxday"]:
                        bonus = 2 if tile["fertilized_until_day"] >= day else 1
                        val = 200 + bonus * price
                    else:
                        # Keep-alive still protects the whole plant, so it must
                        # outrank starting a NEW plant. v1 planted faster than
                        # it could water and lost nearly every crop overnight.
                        val = 180
                tasks.append({"pos": (x, y), "op": ["WATER"], "value": val,
                              "needs": None, "tag": ("tile", x, y)})

            # Harvest when the plant has actually finished growing. For
            # one-time crops harvesting early throws away the bonus window.
            ready = tile["yield_units"] > 0 and age >= cd["first"]
            if not cd["ongoing"]:
                ready = ready and age >= cd["maxday"]
            if ready:
                tasks.append({"pos": (x, y), "op": ["HARVEST"],
                              "value": tile["yield_units"] * price * 2 + 150,
                              "needs": None, "tag": ("tile", x, y)})
            continue

        # ---- animal structures ---------------------------------------------
        if kind in ("COOP", "PASTURE"):
            if "animal" not in tile:
                empty_structs.append((x, y, kind))
                continue

            n_animals += 1
            a = ANIMALS[tile["animal"]]
            price = prices.get(a["product"], 1)
            if tile["animal"] == "GOOSE":
                n_geese += 1

            if not tile["fed_today"]:
                # Same reasoning as watering: an unfed-twice animal escapes and
                # is unrecoverable, taking $300-$500 of capital with it.
                val = 100000 if tile["consecutive_unfed"] >= 1 else 300
                tasks.append({"pos": (x, y), "op": ["FEED"], "value": val,
                              "needs": "WHEAT", "tag": ("tile", x, y)})

            if tile["yield_units"] > 0:
                tasks.append({"pos": (x, y), "op": ["HARVEST"],
                              "value": tile["yield_units"] * price * 2 + 150,
                              "needs": None, "tag": ("tile", x, y)})

            if tile.get("fertilizer_available"):
                # One action for one fertilizer, worth ~$100 early. This is the
                # single best action-for-coins trade in the opening game.
                tasks.append({"pos": (x, y), "op": ["COLLECT_FERTILIZER"],
                              "value": prices.get("FERTILIZER", 1),
                              "needs": None, "tag": ("tile", x, y)})

            if not tile.get("cared_today"):
                # CARE banks +1 on the next production tick, so for a goose
                # (daily production) it straight up doubles output.
                tasks.append({"pos": (x, y), "op": ["CARE"], "value": price * 0.9,
                              "needs": None, "tag": ("tile", x, y)})
            continue

    # ---- place a bought animal onto an empty structure ----------------------
    for animal in ("GOOSE", "COW", "SHEEP"):
        held = shed.get(animal, 0)
        if held <= 0:
            continue
        struct = ANIMALS[animal]["structure"]
        for (x, y, kind) in empty_structs:
            if kind == struct:
                tasks.append({"pos": (x, y), "op": ["PLACE", animal],
                              "value": 900, "needs": animal, "tag": ("tile", x, y)})

    # ---- build coops and plant seeds on bare ground -------------------------
    # Coops go on tiles nearest the shed: animals need 4 visits a day, melons
    # only need 1, so short walks should be spent on animals.
    empty_tiles.sort(key=lambda p: dist(p, (board // 2, board // 2)))
    free_coops = len([s for s in empty_structs if s[2] == "COOP"])

    # Labour budget. Each animal costs ~4 actions/day, each crop ~1.3, and
    # roughly 45% of all turns are spent walking. Expanding past this is how v1
    # ended up with 40 empty coops and a field of weeds.
    n_units = 1 + max(len(farm["hands"]), MAX_HANDS - 2)
    capacity = 24 * n_units * 0.52
    load = n_animals * 4.0 + n_plants * 1.3
    room = capacity - load

    # Only build a coop if a goose is already waiting or affordable, AND we
    # have the labour to feed it.
    coop_budget = 0
    if day <= GOOSE_LAST_DAY and room > 8:
        wanted = shed.get("GOOSE", 0) + (1 if farm["money"] > 900 else 0)
        coop_budget = max(0, min(int(room // 4), wanted - free_coops))

    # One PLANT task per seed actually held, otherwise every idle unit walks to
    # a different tile to plant the same single seed.
    melon_seeds = priv["seeds"].get("MELON", 0) if day <= MELON_LAST_DAY else 0
    wheat_seeds = priv["seeds"].get("WHEAT", 0)
    plant_room = max(0, int(room // 1.3))
    # Feed self-sufficiency: a wheat tile yields ~6 per 4 days, a goose eats 1
    # per day, so roughly 1 tile per 1.5 birds. Buying that volume instead
    # would walk the wheat price from $25 to $60.
    wheat_quota = int(math.ceil(n_animals / 1.5)) - n_wheat_tiles
    if wheat_quota > 0:
        melon_seeds = min(melon_seeds, max(0, plant_room - wheat_quota))

    built = planted_m = planted_w = 0
    for (x, y) in empty_tiles:
        if built < coop_budget:
            tasks.append({"pos": (x, y), "op": ["BUILD_COOP"], "value": 250,
                          "needs": None, "tag": ("tile", x, y)})
            built += 1
        elif planted_m < min(melon_seeds, plant_room):
            tasks.append({"pos": (x, y), "op": ["PLANT", "MELON"], "value": 120,
                          "needs": None, "tag": ("tile", x, y)})
            planted_m += 1
        elif planted_w < min(wheat_seeds, plant_room - planted_m):
            # Wheat tiles feed the geese. Buying 1000+ wheat off the market is
            # ruinous (scarcity curve is 25 + sqrt(x)), so we grow it.
            tasks.append({"pos": (x, y), "op": ["PLANT", "WHEAT"], "value": 55,
                          "needs": None, "tag": ("tile", x, y)})
            planted_w += 1

    # SUPPLY tasks. FEED consumes wheat from the UNIT's inventory, so if nobody
    # is carrying any, every animal on the farm starves. This was the single
    # biggest bug in v1: 16 geese placed, 15 lost.
    wheat_needed = sum(1 for t in tasks if t["needs"] == "WHEAT")
    carried = sum(u.get("WHEAT", 0) for u in priv.get("inventories", []))
    if wheat_needed > carried and shed.get("WHEAT", 0) > 0:
        sp_sorted = shed_tiles(board)[:2]
        for i, sp in enumerate(sp_sorted):
            tasks.append({"pos": sp, "op": ["PICKUP", "WHEAT", 10],
                          "value": 500, "needs": None, "tag": ("supply", i)})

    # Courier for animals sitting in the shed waiting for a coop.
    for a in ("GOOSE", "COW", "SHEEP"):
        if shed.get(a, 0) > 0 and any(s[2] == ANIMALS[a]["structure"] for s in empty_structs):
            held = sum(u.get(a, 0) for u in priv.get("inventories", []))
            if held < shed.get(a, 0):
                for i, sp in enumerate(shed_tiles(board)):
                    tasks.append({"pos": sp, "op": ["PICKUP", a, 1],
                                  "value": 1200, "needs": None, "tag": ("carry", a, i)})

    return tasks, n_geese, n_animals, n_plants


# --------------------------------------------------------------------------
# Assignment: match units to tasks, best value-per-turn first
# --------------------------------------------------------------------------
def assign(units, tasks, board, priv, shed_has_wheat, need_wheat):
    """units is a list of (index, (x, y), inventory)."""
    actions = {}
    taken_tags = set()
    free = list(range(len(units)))
    sheds = shed_tiles(board)

    # Score every (unit, task) pair. Boards are small so this is cheap.
    pairs = []
    for ui in free:
        _, upos, uinv = units[ui]
        for ti, t in enumerate(tasks):
            if t["needs"] and uinv.get(t["needs"], 0) <= 0:
                continue  # unit is not carrying the required item
            d = dist(upos, t["pos"])
            pairs.append((t["value"] / (d + 1.0), ui, ti))
    pairs.sort(reverse=True)

    used_units = set()
    for _, ui, ti in pairs:
        if ui in used_units:
            continue
        t = tasks[ti]
        if t["tag"] in taken_tags:
            continue
        used_units.add(ui)
        taken_tags.add(t["tag"])
        _, upos, _ = units[ui]
        actions[ui] = t["op"] if upos == t["pos"] else step_toward(upos, t["pos"])

    # Units with nothing to do go on logistics duty: fetch wheat for feeding,
    # or drop a full inventory so the goods become sellable from the shed.
    for ui in range(len(units)):
        if ui in used_units:
            continue
        _, upos, uinv = units[ui]
        target = min(sheds, key=lambda s: dist(upos, s))
        carrying = inv_count(uinv)

        if carrying >= DROP_INV_AT:
            actions[ui] = ["DROP"] if upos in sheds else step_toward(upos, target)
        elif need_wheat and shed_has_wheat > 0 and uinv.get("WHEAT", 0) < 3:
            actions[ui] = (["PICKUP", "WHEAT", 6] if upos in sheds
                           else step_toward(upos, target))
        elif carrying > 0:
            actions[ui] = ["DROP"] if upos in sheds else step_toward(upos, target)
        else:
            actions[ui] = ["PASS"]

    return actions


# --------------------------------------------------------------------------
# Market policy
# --------------------------------------------------------------------------
def build_market(obs, farm, priv, board, prices, n_geese, load):
    """Market orders are FREE -- they do not consume a farmer action, and we get
    10 per turn. So there is no reason to be shy about selling."""
    day, hour = obs["day"], obs["hour"]
    money = farm["money"]
    shed = priv["shed"]
    orders = []

    # 1) Hire first: cheapest coins-to-actions conversion in the game.
    if hour == 0:
        # Hands cost fib(n): 11 of them is only ~$232/day against a farm
        # turning over thousands. Size the crew to the upkeep burden.
        want = min(MAX_HANDS, max(4, int(math.ceil((load + 16) / 17.0))))
        if day == 0:
            want = 6
        budget = money - CASH_RESERVE
        n = farm["hires_today"]
        a, b = 1, 1
        for _ in range(n):
            a, b = b, a + b
        while len(orders) < 6 and farm["hires_today"] + len(
                [o for o in orders if o[0] == "HIRE"]) < want and a <= budget:
            orders.append(["HIRE"])
            budget -= a
            a, b = b, a + b

    # 2) Sell everything sellable. Prices only fall with cumulative volume and
    #    never recover much, so holding stock gains nothing -- and the opponent
    #    is draining the same curve, so being first matters.
    shed_total = sum(shed.values())
    for item in ("MELON", "EGG", "FERTILIZER", "MILK", "WOOL",
                 "TOMATO", "STRAWBERRY", "CARROT"):
        q = shed.get(item, 0)
        if q <= 0:
            continue
        if item == "FERTILIZER":
            # Keep a little back to fertilize melons, which doubles the daily
            # watering bonus and gets them to max yield two days sooner.
            q = max(0, q - 4)
        if q > 0 and len(orders) < 9:
            orders.append(["SELL", item, q])

    # Only dump wheat if we are over-stocked; it is animal feed first.
    spare_wheat = shed.get("WHEAT", 0) - WHEAT_BUFFER - n_geese * 2
    if spare_wheat > 0 and len(orders) < 9:
        orders.append(["SELL", "WHEAT", spare_wheat])

    spend = money - CASH_RESERVE

    # 3) Emergency feed: never let an animal starve for want of $30 of wheat.
    need = n_geese * 3 + 12
    wprice = prices.get("WHEAT", 25)
    if hour % 8 == 0 and shed.get("WHEAT", 0) < need and spend > 300 and len(orders) < 10:
        # Buying at scale walks the scarcity curve (25 + sqrt(drained)) straight
        # up. Cap how hard we are willing to chase it; grown wheat is free.
        cap = 30 if wprice <= 32 else (14 if wprice <= 45 else 4)
        orders.append(["BUY_PRODUCT", "WHEAT", min(cap, need - shed.get("WHEAT", 0))])
        spend -= cap * wprice

    # 4) Land. More tiles is more geese is more uncapped egg revenue.
    n_extra = len(farm["unlocked_quadrants"]) - 1
    if n_extra < 3 and day <= 22 and len(orders) < 10:
        cost = LAND_PRICES[n_extra]
        if spend >= cost + 600:
            orders.append(["BUY_LAND"])
            spend -= cost

    # 5) Geese. $300 for ~$75/day forever; payback in 4-5 days.
    goose_floor = 900 if day <= 8 else 0   # protect the opening melon budget
    feed_stock = shed.get("WHEAT", 0)
    # A goose eats 1 wheat/day and dies after two missed days. Buying birds we
    # have no feed for is a pure $300 write-off, so require real headroom.
    feed_headroom = feed_stock - 2 * n_geese
    if (day <= GOOSE_LAST_DAY and len(orders) < 10
            and shed_total < SHED_SOFT_CAP and feed_headroom >= 4):
        afford = int((spend - goose_floor) // 300)
        n = max(0, min(afford, 2, feed_headroom // 4))
        if n >= 1:
            orders.append(["BUY_ANIMAL", "GOOSE", n])
            spend -= 300 * n

    # 6) Seeds. Melon is the best coins-per-action crop in the game; wheat is
    #    bought only as feed backup for the geese.
    if day <= MELON_LAST_DAY and len(orders) < 10:
        have = priv["seeds"].get("MELON", 0)
        want = 14 if day <= 12 else 6
        if have < want and spend >= 80:
            n = min(want - have, int(spend // 80))
            if n > 0:
                orders.append(["BUY_SEED", "MELON", n])
                spend -= 80 * n

    if len(orders) < 10 and n_geese >= 2 and day <= 26:
        have_w = priv["seeds"].get("WHEAT", 0)
        want_w = min(20, max(6, int(n_geese * 0.8)))
        if have_w < want_w and spend >= 10:
            n = min(want_w - have_w, int(spend // 10))
            if n > 0:
                orders.append(["BUY_SEED", "WHEAT", n])

    return orders[:10]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def agent(obs, config=None):
    # A crash is a loss, so the whole brain sits inside a try/except and falls
    # back to a legal no-op.
    try:
        return _think(obs)
    except Exception:
        n = 0
        try:
            n = len(obs["farms"][obs["player"]]["hands"])
        except Exception:
            pass
        return {"farmer": ["PASS"], "hands": [["PASS"]] * n, "market": []}


def _think(obs):
    me = obs["player"]
    farm = obs["farms"][me]
    priv = obs["private"]
    board = len(farm["tiles"])
    prices = obs["market"]["prices"]

    # Unit roster: index 0 is the main farmer, the rest are today's hands.
    invs = priv.get("inventories", [])
    units = [(0, tuple(farm["farmer"]), dict(invs[0]) if invs else {})]
    for i, pos in enumerate(farm["hands"]):
        iv = dict(invs[i + 1]) if len(invs) > i + 1 else {}
        units.append((i + 1, tuple(pos), iv))

    tasks, n_geese, n_animals, n_plants = build_tasks(obs, farm, priv, board, prices)

    # Does anyone still need wheat in hand to feed an animal?
    need_wheat = any(t["needs"] == "WHEAT" for t in tasks)
    # Animals sitting in the shed also need a courier.
    for a in ("GOOSE", "COW", "SHEEP"):
        if priv["shed"].get(a, 0) > 0:
            for t in tasks:
                if t["needs"] == a:
                    need_wheat = need_wheat  # handled by pickup fallback below
                    break

    acts = assign(units, tasks, board, priv, priv["shed"].get("WHEAT", 0), need_wheat)

    load = n_animals * 4.0 + n_plants * 1.2
    market = build_market(obs, farm, priv, board, prices, n_geese, load)

    return {
        "farmer": acts.get(0, ["PASS"]),
        "hands": [acts.get(i, ["PASS"]) for i in range(1, len(units))],
        "market": market,
    }