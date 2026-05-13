from __future__ import annotations

import argparse
import json
import os
from importlib import resources
from pathlib import Path
from typing import Any


def _builtin_skills() -> dict[str, dict[str, Any]]:
    try:
        data = resources.files("scischolar").joinpath("builtin_skills.json").read_text(encoding="utf-8")
        items = json.loads(data)
    except Exception:
        items = []
    return {str(x["name"]): x for x in items if isinstance(x, dict) and x.get("name") and x.get("command")}


def _dirs() -> list[Path]:
    dirs = [Path.cwd() / "skills", Path.home() / ".scischolar" / "skills"]
    extra = os.getenv("SCISCHOLAR_SKILLS_DIR", "")
    dirs += [Path(x).expanduser() for x in extra.split(os.pathsep) if x.strip()]
    return dirs


def _load_user_skills() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for d in _dirs():
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if obj.get("name") and obj.get("command"):
                    obj["_path"] = str(p)
                    out[str(obj["name"])] = obj
            except Exception as exc:
                print(f"Warning: failed to load {p}: {exc}")
    return out


def load_skills() -> dict[str, dict[str, Any]]:
    skills = _builtin_skills()
    skills.update(_load_user_skills())
    aliases = {}
    for name, skill in list(skills.items()):
        for a in skill.get("aliases", []) or []:
            aliases[str(a)] = name
    for a, name in aliases.items():
        if name in skills:
            skills[a] = skills[name]
    return skills


def _flag(k: str) -> str:
    return "--" + k.replace("_", "-")


def _as_argv(defaults: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for k, v in (defaults or {}).items():
        if v is None or v is False:
            continue
        if v is True:
            argv.append(_flag(k))
        elif isinstance(v, list):
            for item in v:
                argv += [_flag(k), str(item)]
        else:
            argv += [_flag(k), str(v)]
    return argv


def expand_skill(skill: dict[str, Any], extra: list[str]) -> list[str]:
    return [str(skill["command"]), *_as_argv(skill.get("defaults", {})), *extra]


def cmd_list(skills: dict[str, dict[str, Any]]) -> int:
    unique = {v["name"]: v for v in skills.values()}
    print("Skill                 Command              Description")
    print("--------------------  -------------------  --------------------------------")
    for name in sorted(unique):
        s = unique[name]
        desc = str(s.get("description", "")).replace("\n", " ")
        if len(desc) > 64:
            desc = desc[:61] + "..."
        print(f"{name:<20}  {s.get('command',''):<19}  {desc}")
    return 0


def cmd_show(skills: dict[str, dict[str, Any]], name: str, raw_json: bool) -> int:
    if name not in skills:
        print(f"Unknown skill: {name}")
        return 2
    obj = {k: v for k, v in skills[name].items() if not k.startswith("_")}
    if raw_json:
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        print(f"# {obj['name']}\n")
        print(obj.get("description", ""))
        print(f"\ncommand: {obj['command']}")
        if obj.get("aliases"):
            print("aliases:", ", ".join(obj["aliases"]))
        print("\ndefaults:")
        for k, v in (obj.get("defaults") or {}).items():
            print(f"  {k}: {v}")
        if obj.get("examples"):
            print("\nexamples:")
            for e in obj["examples"]:
                print(f"  {e}")
    return 0


def cmd_init(skills: dict[str, dict[str, Any]], name: str, source: str | None, output: str | None, force: bool) -> int:
    if source:
        if source not in skills:
            print(f"Unknown source skill: {source}")
            return 2
        obj = {k: v for k, v in skills[source].items() if not k.startswith("_")}
        obj["name"] = name
        obj["aliases"] = []
    else:
        obj = {
            "name": name,
            "aliases": [],
            "description": "Custom SciScholar skill.",
            "command": "search-papers",
            "defaults": {
                "retrieval_mode": "hybrid",
                "top_k": 3,
                "top_keywords": 0,
                "max_titles": 0,
                "max_refs": 0,
                "report_max_items": 3
            },
            "examples": [f"scischolar skill run {name} --query \"open world agent\" --keyword \"high:open world agent\""]
        }
    path = Path(output).expanduser() if output else Path.cwd() / "skills" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        print(f"Skill already exists: {path}. Use --force to overwrite.")
        return 1
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Created: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scischolar skill", description="Editable downstream skills for SciScholar CLI.")
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="List skills.")
    sp = sub.add_parser("show", help="Show skill definition.")
    sp.add_argument("name")
    sp.add_argument("--json", action="store_true")
    rp = sub.add_parser("run", help="Run a skill; extra args are forwarded to the underlying command.")
    rp.add_argument("--dry-run", action="store_true")
    rp.add_argument("name")
    rp.add_argument("extra", nargs=argparse.REMAINDER)
    ip = sub.add_parser("init", help="Create a local editable skill JSON.")
    ip.add_argument("name")
    ip.add_argument("--from", dest="source")
    ip.add_argument("--output")
    ip.add_argument("--force", action="store_true")
    sub.add_parser("where", help="Show user skill directories.")
    return p


def dispatch_skill_cli(argv: list[str]) -> int | list[str]:
    args = build_parser().parse_args(argv)
    skills = load_skills()
    if args.action == "list":
        return cmd_list(skills)
    if args.action == "where":
        for d in _dirs():
            print(d)
        return 0
    if args.action == "show":
        return cmd_show(skills, args.name, args.json)
    if args.action == "init":
        return cmd_init(skills, args.name, args.source, args.output, args.force)
    if args.action == "run":
        if args.name not in skills:
            print(f"Unknown skill: {args.name}")
            print("Use `scischolar skill list`.")
            return 2
        expanded = expand_skill(skills[args.name], list(args.extra or []))
        if args.dry_run:
            print("scischolar " + " ".join(expanded))
            return 0
        return expanded
    return 0
