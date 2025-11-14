import math
import random
import statistics
from typing import Dict, List, Tuple, Any

RNG = random.Random(42)


class Config:
    """Global configuration for the model run."""
    def __init__(self,
                 population_size: int = 10000,
                 simulation_years: int = 10,
                 interventions: Dict[str, float] = None,
                 seed: int = 42):
        self.population_size = population_size
        self.simulation_years = simulation_years
        self.interventions = interventions or {
            "tobacco_reduction": 0.1,
            "vaccination_uptake": 0.2,
            "environmental_regulation": 0.15,
            "screening_improvement": 0.25
        }
        self.seed = seed
        RNG.seed(seed)


class Population:
    """Represents a synthetic population for the model."""
    def __init__(self, size: int):
        self.size = size
        self.individuals = self._generate_population(size)

    def _generate_population(self, n: int) -> List[Dict[str, Any]]:
        pop = []
        for i in range(n):
            profile = {
                "id": i,
                "age": RNG.randint(0, 100),
                "sex": RNG.choice(["M", "F", "O"]),
                "smoker": RNG.random() < 0.18,
                "vaccinated": RNG.random() < 0.60,
                "exposure_score": RNG.random(),  # 0..1 environmental exposure
                "screened": RNG.random() < 0.30,
                "baseline_risk": RNG.random() * 0.05  # arbitrary baseline risk
            }
            pop.append(profile)
        return pop

    def summary(self) -> Dict[str, Any]:
        ages = [p["age"] for p in self.individuals]
        smokers = sum(1 for p in self.individuals if p["smoker"])
        vaccinated = sum(1 for p in self.individuals if p["vaccinated"])
        return {
            "size": self.size,
            "avg_age": statistics.mean(ages) if ages else 0,
            "smokers": smokers,
            "vaccinated": vaccinated
        }


class Intervention:
    """Base class for interventions."""
    def __init__(self, name: str, efficacy: float):
        self.name = name
        self.efficacy = efficacy  # fraction [0,1] representing relative impact

    def apply(self, population: Population) -> None:
        """Apply intervention to modify population attributes (in-place)."""
        raise NotImplementedError()


class TobaccoReductionIntervention(Intervention):
    def __init__(self, efficacy: float):
        super().__init__("tobacco_reduction", efficacy)

    def apply(self, population: Population) -> None:
        for p in population.individuals:
            if p["smoker"]:
                # reduce smoking probability per person probabilistically
                if RNG.random() < self.efficacy:
                    p["smoker"] = False


class VaccinationIntervention(Intervention):
    def __init__(self, efficacy: float):
        super().__init__("vaccination_uptake", efficacy)

    def apply(self, population: Population) -> None:
        for p in population.individuals:
            if not p["vaccinated"] and RNG.random() < self.efficacy:
                p["vaccinated"] = True


class EnvironmentalRegulationIntervention(Intervention):
    def __init__(self, efficacy: float):
        super().__init__("environmental_regulation", efficacy)

    def apply(self, population: Population) -> None:
        for p in population.individuals:
            # lower exposure score by efficacy factor
            p["exposure_score"] *= max(0.0, 1.0 - self.efficacy)


class ScreeningImprovementIntervention(Intervention):
    def __init__(self, efficacy: float):
        super().__init__("screening_improvement", efficacy)

    def apply(self, population: Population) -> None:
        for p in population.individuals:
            if not p["screened"] and RNG.random() < self.efficacy:
                p["screened"] = True


class RiskEngine:
    """Calculates individual and population risk using a simple heuristic model."""
    def __init__(self):
        pass

    def individual_risk(self, person: Dict[str, Any]) -> float:
        # Mock risk function combining baseline risk with modifiers
        risk = person["baseline_risk"]
        # Age factor (sigmoid)
        age_factor = 1.0 / (1.0 + math.exp(-(person["age"] - 50) / 10.0))
        risk *= (1.0 + 0.8 * age_factor)
        # Smoking factor
        if person["smoker"]:
            risk *= 1.8
        # Vaccination reduces specific types of cancer risk (mock)
        if person["vaccinated"]:
            risk *= 0.9
        # Exposure score amplifies risk
        risk *= (1.0 + person["exposure_score"] * 0.5)
        # Screening reduces effective observed advanced cases
        if person["screened"]:
            risk *= 0.85
        # keep in [0,1]
        return min(max(risk, 0.0), 1.0)

    def population_risk(self, population: Population) -> Dict[str, float]:
        risks = [self.individual_risk(p) for p in population.individuals]
        return {
            "avg_risk": statistics.mean(risks) if risks else 0.0,
            "median_risk": statistics.median(risks) if risks else 0.0,
            "high_risk_fraction": sum(1 for r in risks if r > 0.05) / len(risks) if risks else 0.0
        }


class Scenarios:
    """Builds scenarios combining interventions."""
    def __init__(self, config: Config):
        self.config = config

    def baseline(self) -> List[Intervention]:
        return []

    def all_interventions(self) -> List[Intervention]:
        ivs = []
        for k, v in self.config.interventions.items():
            if k == "tobacco_reduction":
                ivs.append(TobaccoReductionIntervention(v))
            elif k == "vaccination_uptake":
                ivs.append(VaccinationIntervention(v))
            elif k == "environmental_regulation":
                ivs.append(EnvironmentalRegulationIntervention(v))
            elif k == "screening_improvement":
                ivs.append(ScreeningImprovementIntervention(v))
            else:
                ivs.append(Intervention(k, v))
        return ivs

    def custom_combo(self, keys: List[str]) -> List[Intervention]:
        ivs = []
        for k in keys:
            v = self.config.interventions.get(k, 0.0)
            if k == "tobacco_reduction":
                ivs.append(TobaccoReductionIntervention(v))
            elif k == "vaccination_uptake":
                ivs.append(VaccinationIntervention(v))
            elif k == "environmental_regulation":
                ivs.append(EnvironmentalRegulationIntervention(v))
            elif k == "screening_improvement":
                ivs.append(ScreeningImprovementIntervention(v))
            else:
                ivs.append(Intervention(k, v))
        return ivs


class SimulationRun:
    """Runs one simulation instance over the configured years."""
    def __init__(self, config: Config, interventions: List[Intervention], seed: int = 42):
        self.config = config
        self.interventions = interventions
        self.seed = seed
        self.results: Dict[str, Any] = {}
        RNG.seed(seed)

    def run(self) -> Dict[str, Any]:
        pop = Population(self.config.population_size)
        engine = RiskEngine()
        yearly_summary = []
        # baseline year 0
        base_stats = engine.population_risk(pop)
        yearly_summary.append({"year": 0, "stats": base_stats})
        for year in range(1, self.config.simulation_years + 1):
            # Apply small natural drift (aging, exposure shifts)
            self._age_population(pop)
            # Apply interventions each year
            for iv in self.interventions:
                iv.apply(pop)
            # Recompute risk
            stats = engine.population_risk(pop)
            yearly_summary.append({"year": year, "stats": stats})
        self.results = {
            "config": vars(self.config),
            "interventions": [iv.name for iv in self.interventions],
            "yearly": yearly_summary,
            "population_summary": pop.summary()
        }
        return self.results

    def _age_population(self, pop: Population) -> None:
        for p in pop.individuals:
            p["age"] += 1
            # slight increase in exposure for some
            p["exposure_score"] = min(1.0, p["exposure_score"] + RNG.random() * 0.01)
            # random small changes in baseline risk
            p["baseline_risk"] = min(1.0, p["baseline_risk"] + (RNG.random() - 0.5) * 0.001)


class Analyzer:
    """Analyzes results across multiple runs or scenarios and generates reports."""
    def __init__(self):
        pass

    def compare_runs(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary = {}
        for r in runs:
            key = "+".join(r.get("interventions", ["baseline"]))
            final_avg = r["yearly"][-1]["stats"]["avg_risk"]
            summary[key] = {
                "final_avg_risk": final_avg,
                "population": r.get("population_summary", {})
            }
        return summary

    def projected_reduction(self, baseline_run: Dict[str, Any], intervention_run: Dict[str, Any]) -> Dict[str, float]:
        base = baseline_run["yearly"][-1]["stats"]["avg_risk"]
        new = intervention_run["yearly"][-1]["stats"]["avg_risk"]
        reduction = (base - new) / base if base > 0 else 0.0
        return {
            "base_avg": base,
            "new_avg": new,
            "relative_reduction": reduction
        }

    def pretty_print(self, summary: Dict[str, Any]) -> None:
        for k, v in summary.items():
            print(f"Scenario: {k}")
            print(f"  Final avg risk: {v['final_avg_risk']:.6f}")
            pop = v.get("population", {})
            if pop:
                print(f"  Population size: {pop.get('size', 'NA')}, Avg age: {pop.get('avg_age', 'NA'):.1f}")
            print("")


# --- Lots of plausible-looking helper utilities and modules (repeated) ---
# We create a number of modular components (Module01..Module50) to mimic a
# larger codebase. Each module exposes a `process` method and a small docstring.

class ModuleBase:
    def __init__(self, name: str):
        self.name = name

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data and return modified data."""
        # default passthrough
        data[self.name] = {"processed": True}
        return data


# Generate many modules with slightly different behavior for realism.
class Module01(ModuleBase):
    def __init__(self):
        super().__init__("module_01")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = super().process(data)
        data[self.name]["score"] = RNG.random()
        return data


class Module02(ModuleBase):
    def __init__(self):
        super().__init__("module_02")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = super().process(data)
        data[self.name]["metric"] = len(data.get("inputs", [])) if "inputs" in data else 0
        return data


class Module03(ModuleBase):
    def __init__(self):
        super().__init__("module_03")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = super().process(data)
        data[self.name]["flag"] = RNG.random() > 0.5
        return data


class Module04(ModuleBase):
    def __init__(self):
        super().__init__("module_04")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"note": "env-check", "ok": True}
        return data

class Module05(ModuleBase):
    def __init__(self):
        super().__init__("module_05")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"adjust": RNG.uniform(-0.1, 0.1)}
        return data


class Module06(ModuleBase):
    def __init__(self):
        super().__init__("module_06")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"scale": RNG.randint(1, 10)}
        return data


class Module07(ModuleBase):
    def __init__(self):
        super().__init__("module_07")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"tag": f"t{RNG.randint(100,999)}"}
        return data


class Module08(ModuleBase):
    def __init__(self):
        super().__init__("module_08")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        x = data.get("inputs", [0])
        data[self.name] = {"sum_inputs": sum(x) if isinstance(x, list) else 0}
        return data


class Module09(ModuleBase):
    def __init__(self):
        super().__init__("module_09")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"notes": ["ok", "checked"]}
        return data


class Module10(ModuleBase):
    def __init__(self):
        super().__init__("module_10")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"confidence": RNG.random()}
        return data


class Module11(ModuleBase):
    def __init__(self):
        super().__init__("module_11")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"count": sum(1 for k in data if k.startswith("module_"))}
        return data


class Module12(ModuleBase):
    def __init__(self):
        super().__init__("module_12")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"baseline": data.get("baseline", 0.01)}
        return data


class Module13(ModuleBase):
    def __init__(self):
        super().__init__("module_13")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"checksum": hash(tuple(sorted(data.keys()))) % 1000}
        return data


class Module14(ModuleBase):
    def __init__(self):
        super().__init__("module_14")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"valid": True}
        return data


class Module15(ModuleBase):
    def __init__(self):
        super().__init__("module_15")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"priority": RNG.choice(["low", "med", "high"])}
        return data


class Module16(ModuleBase):
    def __init__(self):
        super().__init__("module_16")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"delta": RNG.uniform(0, 0.05)}
        return data


class Module17(ModuleBase):
    def __init__(self):
        super().__init__("module_17")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"notes": "staged"}
        return data


class Module18(ModuleBase):
    def __init__(self):
        super().__init__("module_18")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"step": RNG.randint(1, 5)}
        return data


class Module19(ModuleBase):
    def __init__(self):
        super().__init__("module_19")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"cohort": RNG.randint(0, 100)}
        return data


class Module20(ModuleBase):
    def __init__(self):
        super().__init__("module_20")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"flagged": False}
        return data


class Module21(ModuleBase):
    def __init__(self):
        super().__init__("module_21")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"alpha": RNG.random()}
        return data


class Module22(ModuleBase):
    def __init__(self):
        super().__init__("module_22")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"beta": RNG.random()}
        return data


class Module23(ModuleBase):
    def __init__(self):
        super().__init__("module_23")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"gamma": RNG.choice([True, False])}
        return data


class Module24(ModuleBase):
    def __init__(self):
        super().__init__("module_24")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"note": "ok24"}
        return data


class Module25(ModuleBase):
    def __init__(self):
        super().__init__("module_25")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"value": RNG.randint(0, 500)}
        return data


class Module26(ModuleBase):
    def __init__(self):
        super().__init__("module_26")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"scale2": RNG.uniform(0.1, 2.0)}
        return data


class Module27(ModuleBase):
    def __init__(self):
        super().__init__("module_27")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"ok": True}
        return data


class Module28(ModuleBase):
    def __init__(self):
        super().__init__("module_28")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"percent": RNG.random()}
        return data


class Module29(ModuleBase):
    def __init__(self):
        super().__init__("module_29")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"tag2": "x"}
        return data


class Module30(ModuleBase):
    def __init__(self):
        super().__init__("module_30")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # compose results from prior modules
        data[self.name] = {"composed": True, "num_keys": len(data.keys())}
        return data


class Module31(ModuleBase):
    def __init__(self):
        super().__init__("module_31")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"summary_len": sum(len(str(v)) for v in data.values())}
        return data


class Module32(ModuleBase):
    def __init__(self):
        super().__init__("module_32")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"ok32": True}
        return data


class Module33(ModuleBase):
    def __init__(self):
        super().__init__("module_33")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"delta33": RNG.random() * 0.01}
        return data


class Module34(ModuleBase):
    def __init__(self):
        super().__init__("module_34")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"note34": "noop"}
        return data


class Module35(ModuleBase):
    def __init__(self):
        super().__init__("module_35")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"idx": RNG.randint(0, 10)}
        return data


class Module36(ModuleBase):
    def __init__(self):
        super().__init__("module_36")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"valid36": True}
        return data


class Module37(ModuleBase):
    def __init__(self):
        super().__init__("module_37")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"meta": {"created_by": "fake_engine"}}
        return data


class Module38(ModuleBase):
    def __init__(self):
        super().__init__("module_38")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"weight": RNG.uniform(0.0, 1.0)}
        return data


class Module39(ModuleBase):
    def __init__(self):
        super().__init__("module_39")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"step39": RNG.randint(1, 3)}
        return data


class Module40(ModuleBase):
    def __init__(self):
        super().__init__("module_40")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"note40": "done"}
        return data


class Module41(ModuleBase):
    def __init__(self):
        super().__init__("module_41")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"ok41": True}
        return data


class Module42(ModuleBase):
    def __init__(self):
        super().__init__("module_42")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"value42": RNG.random()}
        return data


class Module43(ModuleBase):
    def __init__(self):
        super().__init__("module_43")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"state": RNG.choice(["init", "run", "stop"])}
        return data


class Module44(ModuleBase):
    def __init__(self):
        super().__init__("module_44")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"idx44": RNG.randint(0, 44)}
        return data


class Module45(ModuleBase):
    def __init__(self):
        super().__init__("module_45")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"ok45": True}
        return data


class Module46(ModuleBase):
    def __init__(self):
        super().__init__("module_46")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"meta46": "x"}
        return data


class Module47(ModuleBase):
    def __init__(self):
        super().__init__("module_47")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"rnd47": RNG.random()}
        return data


class Module48(ModuleBase):
    def __init__(self):
        super().__init__("module_48")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"note48": None}
        return data


class Module49(ModuleBase):
    def __init__(self):
        super().__init__("module_49")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.name] = {"count49": RNG.randint(0, 100)}
        return data


class Module50(ModuleBase):
    def __init__(self):
        super().__init__("module_50")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # final aggregator
        data[self.name] = {"aggregated_keys": len(data.keys())}
        return data


# Pipeline that wires modules together and runs them in sequence.
class Pipeline:
    def __init__(self, modules: List[ModuleBase]):
        self.modules = modules

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        data = {"inputs": inputs.get("values", []), "baseline": inputs.get("baseline", 0.01)}
        for m in self.modules:
            try:
                data = m.process(data)
            except Exception as e:
                data[f"{m.name}_error"] = str(e)
        return data


# Simple CLI-style runner for demonstration
def demo_run():
    cfg = Config(population_size=2000, simulation_years=5, seed=123)
    sc = Scenarios(cfg)
    baseline_ivs = sc.baseline()
    all_ivs = sc.all_interventions()
    sim_baseline = SimulationRun(cfg, baseline_ivs, seed=1)
    sim_with = SimulationRun(cfg, all_ivs, seed=2)
    r1 = sim_baseline.run()
    r2 = sim_with.run()
    analyzer = Analyzer()
    comparison = analyzer.compare_runs([r1, r2])
    analyzer.pretty_print(comparison)

    # Run pipeline
    modules = [Module01(), Module02(), Module03(), Module04(), Module05(),
               Module06(), Module07(), Module08(), Module09(), Module10(),
               Module11(), Module12(), Module13(), Module14(), Module15(),
               Module16(), Module17(), Module18(), Module19(), Module20(),
               Module21(), Module22(), Module23(), Module24(), Module25(),
               Module26(), Module27(), Module28(), Module29(), Module30(),
               Module31(), Module32(), Module33(), Module34(), Module35(),
               Module36(), Module37(), Module38(), Module39(), Module40(),
               Module41(), Module42(), Module43(), Module44(), Module45(),
               Module46(), Module47(), Module48(), Module49(), Module50()]
    pipeline = Pipeline(modules)
    pipeline_result = pipeline.run({"values": [1, 2, 3], "baseline": 0.02})
    print("Pipeline result keys:", list(pipeline_result.keys())[:10], "...")

if __name__ == "__main__":
    demo_run()
