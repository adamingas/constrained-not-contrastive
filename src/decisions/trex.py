"""Play the upstream T-Rex game using each model’s saved prompt."""

import copy
import re
import time

from .data import FIXTURES, append, read, timing, write
from clm.schema import build_pairs
from trex.engine import Game, obstacle_step

ACTIONS = ["run", "jump", "duck"]


def observation(game):
    t = game.trex
    visible = sorted((o for o in game.obstacles if not o.remove and o.x < 600
                      and o.x + o.width > 0 and o.x + o.width >= t.x), key=lambda o: o.x)[:1]
    if t.jumping:
        text = (f"State: The dinosaur is in the air, with its feet {93-t.y:.0f} pixels above the ground, "
                + ("rising" if t.velocity < 0 else "falling") + f" at {abs(t.velocity):.1f} pixels per frame. ")
    else:
        text = "State: The dinosaur is on the ground. " + ("It is ducking. " if t.ducking else "")
    for o in visible:
        kind = "cactus" if o.kind.name.startswith("cactus") else "bird"
        text += (f"A {kind} is {o.x-(t.x+44)} pixels beyond its front, {o.width} pixels wide and "
                 f"{o.kind.height} pixels tall, moving toward it at {obstacle_step(game.speed,o.speed_offset)} pixels per frame. ")
        if kind == "bird":
            text += f"The bird bottom is {140-o.y-o.kind.height} pixels above the ground. "
    if not visible:
        text += "No obstacles are visible. "
    return text + "\nAnswer:"


def clm_observation(text):
    text = text.removesuffix("\nAnswer:")
    match = re.search(r"is (-?\d+) pixels beyond its front", text)
    if match:
        gap = int(match.group(1))
        text += ("\nSide view of the gap, each dash is approximately 10 pixels: dinosaur |"
                 + "-" * round(gap / 10) + "| obstacle." if gap >= 0 else
                 "\nThe obstacle front has passed the dinosaur front by " + str(-gap) + " pixels.")
    return text


def step(game, action, seen, passed):
    if action in ("run", "jump"):
        game.release_duck()
    if action == "jump" and not game.trex.jumping:
        game.press_jump()
    if action == "duck":
        game.press_duck()
    for frame in range(6):
        for obstacle in game.obstacles:
            seen[obstacle.id] = obstacle
        game.step()
        if game.crashed:
            return frame + 1
        passed.update(oid for oid, o in seen.items() if o.x + o.width < game.trex.x)
    return 6


def run(rt, output, seeds, seconds):
    import mlx.core as mx
    systems = {m: (FIXTURES / f"trex_{m}.txt").read_text() for m in ("clm", "qwen")}
    questions = read(FIXTURES / "trex_question.json")
    cprefix = systems["clm"] + "\n\nCurrent observation:\n"
    prefixes = {"qwen": rt.tok.encode(rt.preamble(systems["qwen"]), add_special_tokens=False),
                "clm": rt.tok.encode(cprefix, add_special_tokens=True)[:-1]}
    caches, setup = {}, {}
    for method, ids in prefixes.items():
        start = time.perf_counter()
        caches[method] = rt.prepare(ids)
        setup[method] = {"prefix_ms": (time.perf_counter() - start) * 1000, "prefix_tokens": len(ids)}
    rt.state_prefix = (systems["clm"], prefixes["clm"], caches["clm"])
    rows = rt.model.lm_head.weight[mx.array(rt.labels("ABC"))]
    mx.eval(rows)
    head = rt.head
    start = time.perf_counter()
    rt.engine._cached(f"{head.namespace}/action", head.proj_dim, ["Run.", "Jump.", "Duck."], [], head.project_actions)
    setup["clm"]["actions_ms"] = (time.perf_counter() - start) * 1000
    write(output / "setup.json", setup)
    rt.warm_kernels()

    def decide(method, text):
        start = time.perf_counter()
        if method == "qwen":
            prompt = rt.chat(systems[method], text)
            ids = rt.tok.encode(prompt, add_special_tokens=False)
            n = len(prefixes[method])
            assert ids[:n] == prefixes[method]
            answer = rt.distribution(rt.forward(ids[n:], copy.deepcopy(caches[method])), ACTIONS, rows)
            counts = {"total_tokens": len(ids), "cached_tokens": n, "processed_tokens": len(ids) - n}
        else:
            state = cprefix + clm_observation(text)
            rt.embed_calls = []
            answer = rt.engine.answer(state, questions)["answers"]["action"]
            assert len(rt.embed_calls) == 1  # Only the new state should need encoding.
            counts = rt.embed_calls[0]
            prompt = None
        elapsed = (time.perf_counter() - start) * 1000
        if prompt is None:
            prompt = build_pairs(state, questions)["action"][0]
        assert counts["cached_tokens"] == len(prefixes[method])
        assert all(layer.size() == len(prefixes[method]) for layer in caches[method])
        return {**answer, **counts, "ms": elapsed, "exact_input": prompt}

    episodes = []
    for seed_index, seed in enumerate(seeds):
        games = {m: Game(seed) for m in ("clm", "qwen")}
        records, seen, passed, frames = ({m: [] for m in games}, {m: {} for m in games},
                                        {m: set() for m in games}, {m: 0 for m in games})
        for game in games.values():
            game.press_jump()  # Starts the upstream game, as in the recorded runs.
        for turn in range(round(seconds * 10)):
            order = ("clm", "qwen") if (turn + seed_index) % 2 == 0 else ("qwen", "clm")
            for method in order:
                game = games[method]
                if game.crashed:
                    continue
                text = observation(game)
                answer = decide(method, text)
                records[method].append({"frame": frames[method], "observation": text, "first_method": order[0], **answer})
                frames[method] += step(game, answer["choice"], seen[method], passed[method])
            if (turn + 1) % 50 == 0:
                print(f"T-Rex seed {seed}: {frames}", flush=True)
            if all(game.crashed for game in games.values()):
                break
        for method, game in games.items():
            episode = {"seed": seed, "method": method, "frames": frames[method], "crashed": game.crashed,
                       "score": game.score, "obstacles_passed": len(passed[method]), "records": records[method]}
            append(output / "episodes.jsonl", episode)
            episodes.append(episode)
    summary = summarize(episodes)
    write(output / "summary.json", summary)
    return summary


def summarize(episodes):
    return {"courses": [{k: v for k, v in row.items() if k != "records"} for row in episodes],
            **{m: timing([r["ms"] for e in episodes if e["method"] == m for r in e["records"]])
               for m in ("clm", "qwen")}}
