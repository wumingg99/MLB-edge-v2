"""
Tier 3 Phase 1 — Full V2 model backtest on 2025 MLB games.
Uses point-in-time team/pitcher stats (no look-ahead).
Bullpen uses full-season stats (acceptable for Phase 1).
"""
import json, requests, time, sys
import statistics as stat_lib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '/root/mlb-edge-v2')
from dotenv import load_dotenv
load_dotenv('/root/mlb-edge-v2/.env')

MLB = "https://statsapi.mlb.com/api/v1"
TIER1   = Path("tier1_full_2025.json")
SCH     = Path("tier3_schedule.json")
TEAMLOG = Path("tier3_teamlogs.json")
PITLOG  = Path("tier3_pitcherlogs.json")
OUT     = Path("tier3_predictions.json")

def mlb(path, params=None):
    r = requests.get(f"{MLB}{path}", params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()

def sf(v, d=0.0):
    try: return float(v) if v is not None else d
    except: return d

def parse_ip(s):
    if s is None: return 0.0
    try:
        p = str(s).split(".")
        return int(p[0]) + (int(p[1]) if len(p)>1 else 0)/3.0
    except: return 0.0

def ops_from_stats(stats):
    ab=h=d=t=hr=bb=hbp=sfl=0.0
    for s in stats:
        ab+=sf(s.get("atBats")); h+=sf(s.get("hits"))
        d+=sf(s.get("doubles")); t+=sf(s.get("triples"))
        hr+=sf(s.get("homeRuns")); bb+=sf(s.get("baseOnBalls"))
        hbp+=sf(s.get("hitByPitch")); sfl+=sf(s.get("sacFlies"))
    sg=max(h-d-t-hr,0)
    od=ab+bb+hbp+sfl
    obp=(h+bb+hbp)/od if od>0 else 0.320
    slg=(sg+2*d+3*t+4*hr)/ab if ab>0 else 0.400
    return max(0.4, min(obp+slg, 1.4))

def ema(vals, alpha=0.25, default=4.5):
    if not vals: return default
    c=float(vals[0])
    for v in vals[1:]: c=alpha*float(v)+(1-alpha)*c
    return c

def pitch_metrics(splits):
    outs=er=k=bb=hbp=hr=h=air=0.0
    for s in splits:
        st=s.get("stat",s)
        raw_outs=sf(st.get("outs"))
        outs += raw_outs if raw_outs else parse_ip(st.get("inningsPitched"))*3
        er+=sf(st.get("earnedRuns")); k+=sf(st.get("strikeOuts"))
        bb+=sf(st.get("baseOnBalls")); hbp+=sf(st.get("hitBatsmen"))
        hr+=sf(st.get("homeRuns")); h+=sf(st.get("hits"))
        air+=sf(st.get("airOuts"))
    ip=outs/3.0
    if ip<0.1:
        return {"era":4.50,"fip":4.20,"xfip":4.20,"k9":8.0,"bb9":3.0,"whip":1.30,"ip":0.0}
    FIP_C=3.10; LG_HR_FB=0.12
    era_v=min(er*9/ip,15.0)
    fip_v=min(((13*hr+3*(bb+hbp)-2*k)/ip)+FIP_C,15.0)
    xfip_v=min(((13*(air*LG_HR_FB)+3*(bb+hbp)-2*k)/ip)+FIP_C,15.0)
    return {
        "era":round(era_v,3),"fip":round(fip_v,3),"xfip":round(xfip_v,3),
        "k9":round(k*9/ip,3),"bb9":round(bb*9/ip,3),
        "whip":round((bb+h)/ip,3),"ip":round(ip,1),
    }

# ── 1. Load Tier 1 ───────────────────────────────────────────────
tier1 = json.loads(TIER1.read_text())
dates = sorted(d for d in tier1 if tier1[d].get("games",0)>0)
print(f"Tier 1: {len(dates)} dates | {dates[0]} → {dates[-1]}")

# ── 2. Schedule ──────────────────────────────────────────────────
if SCH.exists():
    sched = json.loads(SCH.read_text())
    print(f"Schedule: {len(sched)} games (cached)")
else:
    print("Fetching 2025 schedule...")
    data = mlb("/schedule", {
        "sportId":1,"gameType":"R",
        "startDate":dates[0],"endDate":dates[-1],
        "hydrate":"team,probablePitcher,venue",
    })
    sched = {}
    for de in data.get("dates",[]):
        gd = de["date"]
        for g in de.get("games",[]):
            ht=g.get("teams",{}).get("home",{})
            at=g.get("teams",{}).get("away",{})
            hn=ht.get("team",{}).get("name","")
            an=at.get("team",{}).get("name","")
            sched[f"{gd}||{an} @ {hn}"] = {
                "home_id":  ht.get("team",{}).get("id"),
                "away_id":  at.get("team",{}).get("id"),
                "home_pitcher_id": ht.get("probablePitcher",{}).get("id"),
                "away_pitcher_id": at.get("probablePitcher",{}).get("id"),
                "venue":    g.get("venue",{}).get("name",""),
                "game_time":g.get("gameDate",""),
                "umpire_id": next((o["official"]["id"] for o in g.get("officials",[])
                    if o.get("officialType")=="Home Plate"), None),
            }
    SCH.write_text(json.dumps(sched))
    print(f"  {len(sched)} games cached")

team_ids = set(); pitcher_ids = set()
for v in sched.values():
    if v.get("home_id"): team_ids.add(v["home_id"])
    if v.get("away_id"): team_ids.add(v["away_id"])
    if v.get("home_pitcher_id"): pitcher_ids.add(v["home_pitcher_id"])
    if v.get("away_pitcher_id"): pitcher_ids.add(v["away_pitcher_id"])
print(f"  {len(team_ids)} teams, {len(pitcher_ids)} pitchers")

# ── 3. Team game logs ─────────────────────────────────────────────
if TEAMLOG.exists():
    team_logs = {int(k):v for k,v in json.loads(TEAMLOG.read_text()).items()}
    print(f"Team logs: {len(team_logs)} (cached)")
else:
    print(f"Fetching team logs ({len(team_ids)} teams)...")
    team_logs = {}
    for i,tid in enumerate(sorted(team_ids)):
        try:
            data = mlb(f"/teams/{tid}/stats",{
                "stats":"gameLog","group":"hitting,pitching",
                "season":2025,"gameType":"R"
            })
            hitting=[]; pitching=[]
            for sg in data.get("stats",[]):
                grp=sg.get("group",{}).get("displayName","")
                sp=sorted(sg.get("splits",[]),key=lambda x:x.get("date",""))
                if grp=="hitting": hitting=sp
                elif grp=="pitching": pitching=sp
            team_logs[tid]={"hitting":hitting,"pitching":pitching}
            time.sleep(0.2)
        except Exception as e:
            team_logs[tid]={"hitting":[],"pitching":[]}
        if (i+1)%10==0: print(f"  {i+1}/{len(team_ids)}")
    TEAMLOG.write_text(json.dumps(team_logs))

# ── 4. Pitcher game logs ──────────────────────────────────────────
if PITLOG.exists():
    pit_logs = {int(k):v for k,v in json.loads(PITLOG.read_text()).items()}
    print(f"Pitcher logs: {len(pit_logs)} (cached)")
else:
    print(f"Fetching pitcher logs ({len(pitcher_ids)} pitchers)...")
    pit_logs = {}
    for i,pid in enumerate(sorted(pitcher_ids)):
        try:
            data = mlb(f"/people/{pid}/stats",{
                "stats":"gameLog","group":"pitching",
                "season":2025,"gameType":"R"
            })
            splits=[]
            for sg in data.get("stats",[]): 
                splits=sorted(sg.get("splits",[]),key=lambda x:x.get("date",""))
                break
            pit_logs[pid]=splits
            time.sleep(0.15)
        except Exception as e:
            pit_logs[pid]=[]
        if (i+1)%25==0: print(f"  {i+1}/{len(pitcher_ids)}")
    PITLOG.write_text(json.dumps(pit_logs))

# ── 5. Point-in-time context builders ────────────────────────────
def team_stats_asof(tid, date_str):
    gl=team_logs.get(int(tid),{"hitting":[],"pitching":[]})
    h=[s for s in gl["hitting"]  if s.get("date","")<date_str]
    p=[s for s in gl["pitching"] if s.get("date","")<date_str]
    if not h: return {"ops":0.720,"rpg":4.5,"ra9":4.5,"team_era":4.50,"win_pct":0.500,"games":0}
    n=len(h)
    rf=[sf(s.get("stat",{}).get("runs")) for s in h]
    ra=[sf(s.get("stat",{}).get("runs")) for s in p]
    tot_ip=sum(parse_ip(s.get("stat",{}).get("inningsPitched")) for s in p)
    tot_er=sum(sf(s.get("stat",{}).get("earnedRuns")) for s in p)
    wins=sum(1 for s in h if s.get("isWin"))
    return {
        "ops":     ops_from_stats([s.get("stat",{}) for s in h]),
        "rpg":     sum(rf)/n,
        "ra9":     sum(ra)/n if ra else 4.5,
        "team_era":min(tot_er*9/tot_ip,15.0) if tot_ip>0 else 4.5,
        "win_pct": wins/n,
        "games":   n,
    }

def team_form_asof(tid, date_str, last_n=10):
    gl=team_logs.get(int(tid),{"hitting":[],"pitching":[]})
    all_h=[s for s in gl["hitting"]  if s.get("date","")<date_str]
    all_p=[s for s in gl["pitching"] if s.get("date","")<date_str]
    h=all_h[-last_n:]; p=all_p[-last_n:]
    if not h: return {"recent_rpg":4.5,"recent_ra9":4.5,"recent_ops":0.720,
                       "recent_win_pct":0.5,"season_win_pct":0.5,"ema_rpg":4.5,"ema_ra9":4.5,"games":0}
    rf=[sf(s.get("stat",{}).get("runs")) for s in h]
    ra=[sf(s.get("stat",{}).get("runs")) for s in p]
    arf=[sf(s.get("stat",{}).get("runs")) for s in all_h]
    ara=[sf(s.get("stat",{}).get("runs")) for s in all_p]
    return {
        "recent_rpg":     sum(rf)/len(rf) if rf else 4.5,
        "recent_ra9":     sum(ra)/len(ra) if ra else 4.5,
        "recent_ops":     ops_from_stats([s.get("stat",{}) for s in h]),
        "recent_win_pct": sum(1 for s in h if s.get("isWin"))/len(h),
        "season_win_pct": sum(1 for s in all_h if s.get("isWin"))/len(all_h) if all_h else 0.5,
        "ema_rpg":        ema(arf[-30:]),
        "ema_ra9":        ema(ara[-30:]),
        "games":          len(all_h),
    }

def pit_stats_asof(pid, date_str, last_n=None):
    if not pid: return {"era":4.50,"fip":4.20,"xfip":4.20,"k9":8.0,"bb9":3.0,"whip":1.30,"ip":0.0}
    sp=[s for s in pit_logs.get(int(pid),[]) if s.get("date","")<date_str]
    if last_n: sp=sp[-last_n:]
    return pitch_metrics(sp) if sp else {"era":4.50,"fip":4.20,"xfip":4.20,"k9":8.0,"bb9":3.0,"whip":1.30,"ip":0.0}


from datetime import datetime as _dt, timedelta as _td

def bullpen_asof(tid, date_str):
    """Approximate bullpen ERA and workload from team pitching logs."""
    gl = team_logs.get(int(tid), {"hitting": [], "pitching": []})
    p  = [s for s in gl["pitching"] if s.get("date","") < date_str]
    if not p: return {"bullpen_era": 4.50, "bullpen_workload": 0.0}
    # Season bullpen ERA: use team ERA + small adjustment (starters typically lower)
    tot_er  = sum(sf(s.get("stat",{}).get("earnedRuns")) for s in p)
    tot_ip  = sum(parse_ip(s.get("stat",{}).get("inningsPitched")) for s in p)
    team_era = min(tot_er * 9 / tot_ip, 15.0) if tot_ip > 0 else 4.50
    bullpen_era = min(team_era * 1.05, 15.0)  # bullpen slightly higher than team avg
    # Workload: bullpen IP in last 3 days (team IP - ~5.5 starter IP per game)
    cutoff = (_dt.strptime(date_str, "%Y-%m-%d") - _td(days=3)).strftime("%Y-%m-%d")
    recent = [s for s in p if s.get("date","") >= cutoff]
    raw_ip  = sum(parse_ip(s.get("stat",{}).get("inningsPitched")) for s in recent)
    workload = max(raw_ip - 5.5 * len(recent), 0.0)
    return {"bullpen_era": round(bullpen_era,2), "bullpen_workload": round(workload,2)}

def rest_days_asof(tid, date_str):
    """Compute team rest days from most recent game in hitting log."""
    gl = team_logs.get(int(tid), {"hitting": []})
    prior = [s.get("date","") for s in gl["hitting"] if s.get("date","") < date_str]
    if not prior: return 3
    last = max(prior)
    delta = (_dt.strptime(date_str, "%Y-%m-%d") - _dt.strptime(last, "%Y-%m-%d")).days
    return min(max(delta, 0), 10)

def build_context(game_info, date_str):
    hi=game_info.get("home_id"); ai=game_info.get("away_id")
    hpi=game_info.get("home_pitcher_id"); api_=game_info.get("away_pitcher_id")
    hs=team_stats_asof(hi,date_str); as_=team_stats_asof(ai,date_str)
    hf=team_form_asof(hi,date_str);  af=team_form_asof(ai,date_str)
    hp=pit_stats_asof(hpi,date_str); ap=pit_stats_asof(api_,date_str)
    hpr={"recent_era":pit_stats_asof(hpi,date_str,3)["era"],
         "recent_k9": pit_stats_asof(hpi,date_str,3)["k9"],
         "recent_bb9":pit_stats_asof(hpi,date_str,3)["bb9"],
         "recent_ip": pit_stats_asof(hpi,date_str,3)["ip"]}
    apr={"recent_era":pit_stats_asof(api_,date_str,3)["era"],
         "recent_k9": pit_stats_asof(api_,date_str,3)["k9"],
         "recent_bb9":pit_stats_asof(api_,date_str,3)["bb9"],
         "recent_ip": pit_stats_asof(api_,date_str,3)["ip"]}
    has_p=bool(hpi and api_ and hp.get("ip",0)>0 and ap.get("ip",0)>0)
    g=hs.get("games",0)
    qual=stat_lib.fmean([
        1.0 if has_p else 0.0,
        min(hp.get("ip",0)/30,1.0), min(ap.get("ip",0)/30,1.0),
        min(g/20,1.0), min(as_.get("games",0)/20,1.0), 0.5
    ])
    return {
        "home_stats":hs,"away_stats":as_,
        "home_form":hf,"away_form":af,
        "home_pitcher":hp,"away_pitcher":ap,
        "home_pitcher_recent":hpr,"away_pitcher_recent":apr,
        "home_bullpen": bullpen_asof(hi, date_str),
        "away_bullpen": bullpen_asof(ai, date_str),
        "park_factor":_PARK.get(game_info.get("venue",""),{"runs":1.0,"hr":1.0})["runs"],
        "park_hr_factor":_PARK.get(game_info.get("venue",""),{"runs":1.0,"hr":1.0})["hr"],
        "weather":get_historical_weather(game_info.get("venue",""), game_info.get("game_time","")), 
        "home_rest_days":1,"away_rest_days":1,
        "has_real_pitchers":has_p,
        "umpire_factor":_UF.get(game_info.get("umpire_id"),1.0),
        "home_lineup_woba":0.320,
        "away_lineup_woba":0.320,
        "data_quality":round(qual,3),
        "venue":game_info.get("venue",""),
        "as_of":f"{date_str}T15:00:00Z",
        "game_date":date_str,
    }

def build_odds(row):
    total = row.get("total_open")
    ov_p  = row.get("over_price")
    un_p  = row.get("under_price")
    if not total or not ov_p or not un_p:
        return None  # skip games with no real prices
    rl      = row.get("home_rl")
    away_rl = row.get("away_rl")
    hp      = row.get("home_rl_price")
    ap      = row.get("away_rl_price")
    return {
        "total":total,"run_line":rl,"over_price":ov_p,"under_price":un_p,
        "home_price":hp,"away_price":ap,
        "total_market":{"point":total,"over_price":ov_p,"under_price":un_p,"books":1},
        "spread_market":{"home_point":rl,"away_point":away_rl,
                         "home_price":hp,"away_price":ap,"books":1},
        "bookmaker_count":1,"quote_timestamp":"2025",
    }

# ── Name normalizer (Odds API → MLB Stats API) ──────────────────
NAME_MAP = {"Oakland Athletics": "Athletics"}
def norm_game(s):
    for o, n in NAME_MAP.items(): s = s.replace(o, n)
    return s

# ── 6. Run model ──────────────────────────────────────────────────
print("\nRunning V2 model on historical games...")
from model import predict_game
from data import get_park_factors as _get_pf
_PARK = _get_pf()
import json as _json_uf
_UF_PATH = Path("/root/mlb-edge-v3/umpire_factors.json")
_UF = {int(k): v["factor"] for k,v in _json_uf.loads(_UF_PATH.read_text()).items()} if _UF_PATH.exists() else {}
print(f"Umpire factors: {len(_UF)} umpires")

# Historical weather cache
_WEATHER_CACHE_PATH = Path("/root/mlb-edge-v2/tier3_weather_2025.json")
_WEATHER = json.loads(_WEATHER_CACHE_PATH.read_text()) if _WEATHER_CACHE_PATH.exists() else {}
print(f"Weather cache: {len(_WEATHER)} venues loaded")


DOMES = {
    "Tropicana Field","Globe Life Field","Chase Field","Minute Maid Park",
    "Daikin Park","American Family Field","loanDepot park","Rogers Centre",
}

def get_historical_weather(venue, game_time):
    if venue in DOMES:
        return {"temp_f":72,"wind_mph":0,"factor":1.0,"is_dome":True,"available":True}
    if not _WEATHER or venue not in _WEATHER:
        return {"temp_f":72,"wind_mph":5,"factor":1.0,"is_dome":False,"available":False}
    date_str = game_time[:10] if game_time else ""
    day = _WEATHER[venue].get(date_str, {})
    if not day:
        return {"temp_f":72,"wind_mph":5,"factor":1.0,"is_dome":False,"available":False}
    for h in [19, 20, 18, 21, 17]:
        if str(h) in day:
            return {**day[str(h)], "available": True}
    return {"temp_f":72,"wind_mph":5,"factor":1.0,"is_dome":False,"available":False}

from model import predict_game

results=[]; skip=0
for date_str, rec in sorted(tier1.items()):
    for row in rec.get("data",[]):
        key=f"{date_str}||{norm_game(row['game'])}"
        gi=sched.get(key)
        if not gi: skip+=1; continue
        ctx=build_context(gi,date_str)
        odds=build_odds(row)
        if not odds: skip+=1; continue
        try:
            pred=predict_game(ctx,total=row.get("total_open"),
                              run_line=row.get("home_rl"),odds_entry=odds)
        except Exception as e:
            skip+=1; continue
        if not pred: skip+=1; continue
        if (len(results)+skip) % 100 == 0:
            print(f"  ...{len(results)+skip} games done ({len(results)} processed)", flush=True)

        tclv=rclv=None
        if pred.get("edge_flagged") and row.get("total_move") is not None:
            tclv=row["total_move"] if pred.get("total_pred")=="OVER" else -row["total_move"]
        if pred.get("rl_edge_flagged") and row.get("rl_open_nv") is not None:
            onv=row["rl_open_nv"]; cnv=row.get("rl_close_nv", onv)
            rclv=round((cnv-onv)*100,2) if pred.get("rl_side")=="HOME" else round((onv-cnv)*100,2)

        results.append({
            "date":date_str,"game":row["game"],
            "total_pred":pred.get("total_pred"),"total_conf":pred.get("total_conf"),
            "total_ev":pred.get("total_ev"),"edge_flagged":pred.get("edge_flagged"),
            "rl_side":pred.get("rl_side"),"rl_conf":pred.get("rl_conf"),
            "rl_ev":pred.get("rl_ev"),"rl_edge_flagged":pred.get("rl_edge_flagged"),
            "total_clv":tclv,"rl_clv":rclv,
            "data_quality":pred.get("data_quality"),
            "total_open":row.get("total_open"),"total_close":row.get("total_close"),
        })

OUT.write_text(json.dumps(results,indent=2))
print(f"  {len(results)} games processed | {skip} skipped → {OUT}")

print(f"\n{'='*62}")
print("TIER 3 RESULTS — V2 MODEL, 2025 MLB, PINNACLE + WEATHER")
print(f"{'='*62}")

edge_t=[r for r in results if r.get("edge_flagged")]
edge_r=[r for r in results if r.get("rl_edge_flagged")]
et_clv=[r["total_clv"] for r in edge_t if r.get("total_clv") is not None]
er_clv=[r["rl_clv"]    for r in edge_r if r.get("rl_clv")   is not None]

print(f"\nGames processed: {len(results)} | Skipped: {skip}")
print(f"Edge total picks:  {len(edge_t)} ({len(et_clv)} with CLV data)")
print(f"Edge RL picks:     {len(edge_r)} ({len(er_clv)} with CLV data)")

if et_clv:
    pos=sum(1 for v in et_clv if v>0)
    print(f"\nTOTAL LINE CLV:")
    print(f"  avg={stat_lib.mean(et_clv):+.3f}pts | median={stat_lib.median(et_clv):+.3f}pts | n={len(et_clv)}")
    print(f"  {pos}/{len(et_clv)} positive ({pos/len(et_clv):.0%})")
    if len(et_clv)>1: print(f"  stdev={stat_lib.stdev(et_clv):.3f}pts")

if er_clv:
    pos=sum(1 for v in er_clv if v>0)
    print(f"\nRUN LINE CLV:")
    print(f"  avg={stat_lib.mean(er_clv):+.2f}pp | median={stat_lib.median(er_clv):+.2f}pp | n={len(er_clv)}")
    print(f"  {pos}/{len(er_clv)} positive ({pos/len(er_clv):.0%})")

bkts=defaultdict(list)
for r in edge_t:
    if r.get("total_clv") is not None:
        bkts[round((r.get("total_conf") or 50)/5)*5].append(r["total_clv"])
if bkts:
    print(f"\nCONFIDENCE vs CLV (totals, 5% buckets):")
    for b in sorted(bkts):
        if len(bkts[b])<3: continue
        vals=bkts[b]; pos=sum(1 for v in vals if v>0)
        print(f"  ~{b}% conf: n={len(vals)} | avg={stat_lib.mean(vals):+.3f}pts | {pos}/{len(vals)} pos")
