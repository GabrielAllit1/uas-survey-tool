# unified_mathtoolbox.py
from __future__ import annotations
import os
import math
import json
import itertools
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union, Callable, Tuple

import numpy as np
import sympy as sp
from sympy.abc import x, y
from sympy import Poly, simplify
from mpmath import mp
from scipy.stats import entropy as scipy_entropy, gumbel_r, norm
from scipy.integrate import quad
from scipy.special import comb

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False
    cp = None

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except Exception:
    NETWORKX_AVAILABLE = False
    nx = None

# Copulas (optional)
try:
    from copulas.multivariate import GaussianMultivariate
    from copulas.bivariate import Gaussian as CopulaGaussian, Clayton as CopulaClayton
    COPULAS_AVAILABLE = True
except Exception:
    COPULAS_AVAILABLE = False
    GaussianMultivariate = None
    CopulaGaussian = None
    CopulaClayton = None

# Gudhi for persistent homology (optional)
try:
    import gudhi as gd
    GUDHI_AVAILABLE = True
except Exception:
    GUDHI_AVAILABLE = False
    gd = None

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logger = logging.getLogger("UnifiedMathToolBox")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------------------------------------------------------------------
# Category-theory helpers
# ----------------------------------------------------------------------------
class Functor:
    """Basic functor class for mapping objects and morphisms (Tool #5228)."""
    def __init__(self, map_obj: Callable[[Any], Any], map_morph: Callable[[Any], Any]):
        self.map_obj = map_obj
        self.map_morph = map_morph

    def apply_object(self, obj: Any) -> Any:
        return self.map_obj(obj)

    def apply_morphism(self, morph: Any) -> Any:
        return self.map_morph(morph)


def compose_functors(f: Functor, g: Functor) -> Functor:
    """Compose two functors (Tool #5229)."""
    return Functor(
        lambda obj: f.apply_object(g.apply_object(obj)),
        lambda m: f.apply_morphism(g.apply_morphism(m))
    )

# ----------------------------------------------------------------------------
# Unified Math Tool Box
# ----------------------------------------------------------------------------
class MathToolBox:
    """
    Unified Math Tool Box consolidating functions from multiple source modules.

    NOTE: accepts and ignores unknown kwargs (e.g. device="cpu") for
    backward-compatibility with older call sites.
    """
    DEFAULT_SCALE_FACTOR: float = 4305.0
    DEFAULT_MODULUS: int = 142857

    def __init__(self, use_cuda: Optional[bool] = None, **kwargs) -> None:
        # Numerical precision
        mp.dps = 80

        # Device selection (ignore unknown kwargs like 'device')
        if use_cuda is None:
            use_cuda = TORCH_AVAILABLE and torch.cuda.is_available()
        self.device = "cuda" if use_cuda else "cpu"

        # Caches
        self._fib_cache: Dict[int, int] = {0: 0, 1: 1}
        self._pi_fib_cache: Dict[int, float] = {0: 0.0, 1: float(mp.pi), 2: float(mp.pi), 3: float(2*mp.pi)}
        self._prime_cache: Dict[int, Tuple[int, ...]] = {}

        logger.info("Math Tool Box initialized on device: %s", self.device)

    # ------------------------------------------------------------------
    # Core sequences / number theory
    # ------------------------------------------------------------------
    @staticmethod
    @lru_cache(maxsize=2048)
    def fibonacci(n: int) -> int:
        if n < 0:
            raise ValueError("n must be non-negative")
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    def unified_detection(self, k: int, t: float, weights: List[float] = None, T: float = 24.0) -> float:
        """
        Composite score in [0,1] capturing recursive, cyclic, fractal, and chaotic signatures.

        Args:
            k: Index of the point being evaluated.
            t: Phase/angle for cyclic cosine term.
            weights: Weights for components [recursive, cyclic, fractal, chaotic].
            T: Period for cyclic component.

        Returns:
            float: Normalized score in [0,1].
        """
        weights = weights if weights is not None else [0.25, 0.25, 0.25, 0.25]
        if len(weights) != 4 or any(w < 0 for w in weights):
            logger.warning(f"Invalid weights {weights}; using default [0.25, 0.25, 0.25, 0.25]")
            weights = [0.25, 0.25, 0.25, 0.25]

        fib = [self.fibonacci(i) for i in range(max(2 * k, k + 1) + 1)]
        fib_mod9 = [1, 1, 2, 3, 5, 8, 4, 3, 7, 1, 8, 0, 8, 8, 7, 6, 4, 1, 5, 6, 2, 8, 1, 0]
        recursive = fib[k]
        cyclic = fib_mod9[k % 24]
        fractal = fib[2 * fib[k] - 2] if 2 * fib[k] - 2 < len(fib) else 0
        chaotic = 0.5  # Placeholder for logistic map

        raw = (weights[0] * recursive + weights[1] * cyclic + weights[2] * fractal +
               weights[3] * chaotic) * (1 + np.cos(2 * np.pi * t / T))
        max_recursive = fib[k]
        norm = (weights[0] * max_recursive + weights[1] * 8 + weights[2] * 1 + weights[3] * 1)
        score = abs(raw) / norm if norm else 0.0
        return float(min(max(score, 0.0), 1.0))

    @lru_cache(maxsize=2048)
    def pi_fibonacci_enhanced(self, n: int) -> mp.mpf:
        if n == 0: return mp.mpf(0)
        if n == 1 or n == 2: return mp.pi()
        a, b = mp.pi(), mp.pi()
        for _ in range(3, n + 1):
            a, b = b, a + b
        return b

    @staticmethod
    @lru_cache(maxsize=2048)
    def tribonacci(n: int) -> int:
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0: return 0
        if n == 1 or n == 2: return 1
        t0, t1, t2 = 0, 1, 1
        for _ in range(3, n + 1):
            t0, t1, t2 = t1, t2, t0 + t1 + t2
        return t2

    @staticmethod
    @lru_cache(maxsize=2048)
    def pell(n: int) -> int:
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0: return 0
        if n == 1: return 1
        a, b = 0, 1
        for _ in range(1, n + 1):
            a, b = b, 2*b + a
        return a

    @staticmethod
    @lru_cache(maxsize=2048)
    def lucas(n: int) -> int:
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0: return 2
        if n == 1: return 1
        a, b = 2, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    @staticmethod
    def golden_ratio_approximation(n: int) -> float:
        phi = (1 + math.sqrt(5)) / 2.0
        return float((phi**n) / math.sqrt(5))

    @staticmethod
    def digital_root(x_val: Union[int, mp.mpf]) -> int:
        if isinstance(x_val, mp.mpf):
            x_int = int(mp.floor(x_val))
        else:
            x_int = int(x_val)
        if x_int == 0:
            return 0
        return x_int % 9 or 9

    @staticmethod
    def p_adic_valuation(x_val: int, p: int = 3) -> Union[int, float]:
        if p <= 1:
            raise ValueError("p must be prime > 1")
        if x_val == 0:
            return float('inf')
        k = 0
        while x_val % p == 0:
            x_val //= p
            k += 1
        return k

    @staticmethod
    def kaprekar_operation(n: int) -> int:
        digits = list(str(n).zfill(4))
        max_perm = int("".join(sorted(digits, reverse=True)))
        min_perm = int("".join(sorted(digits)))
        return max_perm - min_perm

    @staticmethod
    def reptend_length(p: int) -> int:
        if p <= 1 or not sp.isprime(p):
            raise ValueError("p must be prime")
        for k in range(1, p):
            if pow(10, k, p) == 1:
                return k
        return 0

    @staticmethod
    def amicable_pair(n: int) -> int:
        def sum_divisors(x: int) -> int:
            return sum(i for i in range(1, x) if x % i == 0)
        m = sum_divisors(n)
        return n if m != n and sum_divisors(m) == n else 0

    @staticmethod
    def binomial_coefficient(n: int, k: int) -> int:
        if k < 0 or k > n:
            return 0
        return int(comb(n, k, exact=True))

    @staticmethod
    def catalan_number(n: int) -> int:
        return MathToolBox.binomial_coefficient(2*n, n) // (n + 1)

    @staticmethod
    def fibonacci_matrix_element(i: int, j: int) -> int:
        return MathToolBox.fibonacci(i + j - 1)

    # ------------------------------------------------------------------
    # Elliptic curves (heuristic rank)
    # ------------------------------------------------------------------
    def _cached_primes(self, limit: int) -> Tuple[int, ...]:
        if limit in self._prime_cache:
            return self._prime_cache[limit]
        primes = tuple(sp.primerange(2, limit + 1))
        self._prime_cache[limit] = primes
        return primes

    def elliptic_curve_rank_heuristic(self, a: int, b: int, verbose: bool = False) -> float:
        try:
            E = sp.EllipticCurve([a, b])
            disc = E.discriminant()
            _ = abs(disc) ** (1/6)  # Rough conductor proxy
        except Exception:
            pass  # continue with purely local factors

        def ap_coeff(p):
            try:
                pts = 1
                for xv in range(p):
                    rhs = (xv**3 + a*xv + b) % p
                    ls = pow(rhs, (p - 1)//2, p)
                    if rhs == 0:
                        pts += 1
                    elif ls == 1:
                        pts += 2
                return p + 1 - pts
            except Exception:
                return 0

        primes = list(self._cached_primes(557))[:100]

        def L_series(s):
            val = mp.mpf(1.0)
            for p in primes:
                ap = ap_coeff(p)
                term = 1 - ap * p ** (-s) + p ** (1 - 2*s)
                if term == 0:
                    continue
                val *= 1 / term
            return val

        s1 = mp.mpf('1.0')
        delta = mp.mpf('1e-6')
        L1 = L_series(s1)
        Lp = L_series(s1 + delta)
        Lm = L_series(s1 - delta)
        dL = (Lp - Lm) / (2 * delta)
        eps = mp.mpf('1e-5')
        rank = 0
        if abs(L1) < eps:
            rank += 1
            if abs(dL) < eps:
                rank += 1
        if verbose:
            logger.info("L(1)≈%s, dL/ds(1)≈%s, rank≈%d", L1, dL, rank)
        return float(rank)

    # ------------------------------------------------------------------
    # Graph / topology
    # ------------------------------------------------------------------
    @staticmethod
    def euler_characteristic(simplices: List[List[int]]) -> int:
        vertices, edges, faces = set(), set(), set()
        for s in simplices:
            if len(s) == 1: vertices.add(tuple(s))
            elif len(s) == 2: edges.add(tuple(sorted(s)))
            elif len(s) == 3: faces.add(tuple(sorted(s)))
        return len(vertices) - len(edges) + len(faces)

    @staticmethod
    def compute_persistent_homology(points: np.ndarray, max_dim: int = 2, max_edge_length: float = 1.0):
        if not GUDHI_AVAILABLE:
            raise ImportError("Gudhi not available for persistent homology.")
        rips = gd.RipsComplex(points=points, max_edge_length=max_edge_length)
        st = rips.create_simplex_tree(max_dimension=max_dim)
        return st.persistence()

    @staticmethod
    def persistent_homology_betti_numbers(graph, max_dim: int = 2) -> Dict[int, int]:
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required")
        simplices = {}
        simplices[0] = set(frozenset([v]) for v in graph.nodes())
        for dim in range(1, max_dim + 1):
            simplices[dim] = set()
            cliques = list(nx.find_cliques(graph))
            for clique in cliques:
                if len(clique) >= dim + 1:
                    for simplex in itertools.combinations(clique, dim + 1):
                        simplices[dim].add(frozenset(simplex))
        beta_0 = len(simplices[0])
        beta_1 = (len(simplices.get(1, [])) - beta_0 + 1)
        beta_2 = max(0, (len(simplices.get(2, [])) - len(simplices.get(1, [])) + beta_1))
        return {0: beta_0, 1: beta_1, 2: beta_2}

    @staticmethod
    def persistent_entropy(diagram: List[Tuple[float, float]]) -> float:
        persistence = [d - b for b, d in diagram if d > b]
        if not persistence:
            return 0.0
        probs = np.array(persistence, dtype=float)
        probs /= np.sum(probs)
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    # ------------------------------------------------------------------
    # Spectral / entropy / crypto-adjacent
    # ------------------------------------------------------------------
    def zk_oracle_entropy(self, input_array: np.ndarray, use_cuda: Optional[bool] = None) -> float:
        arr = np.asarray(input_array, dtype=np.float32).ravel()
        if use_cuda is None:
            use_cuda = self.device == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available()
        if not use_cuda:
            return self.zk_oracle_entropy_cpu(arr)
        try:
            x = torch.tensor(arr, dtype=torch.float32, device="cuda")
            fft_vals = torch.fft.fft(x)
            psd = torch.abs(fft_vals) ** 2
            psd = psd / (torch.sum(psd) + 1e-20)
            ent = -torch.sum(psd * torch.log2(psd + 1e-20))
            return float(ent.item())
        except Exception as e:
            logger.warning("CUDA entropy failed (%s); falling back to CPU.", e)
            return self.zk_oracle_entropy_cpu(arr)

    @staticmethod
    def zk_oracle_entropy_cpu(input_array: np.ndarray) -> float:
        psd = np.abs(np.fft.fft(input_array)) ** 2
        psd = psd / (np.sum(psd) + 1e-20)
        return float(-np.sum(psd * np.log2(psd + 1e-20)))

    @staticmethod
    def copula_entropy_pairwise(data: np.ndarray) -> float:
        if not COPULAS_AVAILABLE:
            raise ImportError("copulas package required")
        model = GaussianMultivariate()
        model.fit(data)
        samples = model.sample(len(data))
        return float(scipy_entropy(samples.flatten()))

    # ------------------------------------------------------------------
    # Variants / mining heuristics
    # ------------------------------------------------------------------
    @staticmethod
    def kaprekar_nonce(k: int) -> int:
        perms = [int("".join(p)) for p in itertools.permutations(f"{k:04d}")]
        return (max(perms) - min(perms)) % (2**32)

    @staticmethod
    def tropical_kaprekar(k: int) -> int:
        perms = [int("".join(p)) for p in itertools.permutations(f"{k:04d}")]
        return (min(perms) + (max(perms) - min(perms))) % (2**32)

    @staticmethod
    def motivic_entropy_nonce(k: int, cycle_class: float) -> int:
        val = (max(k, 1) * float(cycle_class)) % (2**32)
        return int(val)

    @staticmethod
    def symplectic_hamiltonian_flow(a: float, p: float) -> float:
        H = 0.5 * (p**2) + math.sin(a)
        return float(math.exp(-H))

    @staticmethod
    def compute_graphon_kernel(G) -> float:
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required")
        adj = nx.to_numpy_array(G)
        return float(np.mean(adj))

    @staticmethod
    def extreme_value_gumbel(x: np.ndarray, mu: float, beta: float) -> float:
        return float(np.sum(gumbel_r.pdf(x, loc=mu, scale=beta)))

    @staticmethod
    def stochastic_nonce_step(x_t: int, sigma: float = 1.0) -> int:
        return int((x_t + np.random.normal(0, sigma))) % (2**32)

    @staticmethod
    def ergodic_sequence(k: int, measure: float = 0.99) -> int:
        return int((k * measure) % (2**32))

    @staticmethod
    def martingale_process(a_k: Union[int, float]) -> float:
        x_t = float(a_k)
        x_t1 = x_t + norm.rvs(scale=0.1)
        return abs(x_t1 - x_t)

    @staticmethod
    def bifurcation_analysis(a_k: float) -> float:
        mu = float(a_k)
        x_val = 0.5
        for _ in range(100):
            x_val = mu * x_val * (1 - x_val)
        return abs(x_val)

    @staticmethod
    def zk_oracle_validation(a_k: int, sin_amp: float = 1.0) -> float:
        try:
            return float(sp.nsimplify(sp.log(1 + abs(math.sin(sin_amp * a_k))) * a_k, rational=True))
        except Exception:
            return float(math.log(1 + abs(math.sin(sin_amp * a_k))) * a_k)

    @staticmethod
    def elliptic_zkp_commit(g: int, h: int, m: int, r: int, modulus: int = 97) -> int:
        return (pow(g, m, modulus) * pow(h, r, modulus)) % modulus

    @staticmethod
    def c_star_algebra_norm(a_k: float) -> float:
        return abs(a_k) % 9

    @staticmethod
    def homotopical_consensus(a_k: float) -> float:
        return float(sp.log(1 + abs(a_k))) % 9

    # ------------------------------------------------------------------
    # Prototyped-tools generator
    # ------------------------------------------------------------------
    def generate_prototype_tool(self, tool_id: int, a_k: float, params: Dict[str, float]) -> float:
        s = params.get('s', 1.0)
        h = params.get('h', 1.0)
        sigma = params.get('sigma', 1.0)
        delta_k = params.get('delta_k', 1e-10)
        category = self._get_prototype_category(tool_id)
        if category == "nonce_optimization":
            return float(self.p_adic_valuation(int(a_k)) * abs(math.log(1 + abs(math.sin(a_k * s * h * sigma * delta_k * 1e-20)))) * self.digital_root(a_k)) % 9
        elif category == "consensus_stabilization":
            return min([abs(math.sin(a_k * s)), abs(math.cos(a_k * h))]) % 9
        elif category == "validation":
            return float(self.p_adic_valuation(int(a_k)) * abs(math.log(1 + abs(math.sin(a_k))))) % 9
        else:
            return self.digital_root(a_k * (s ** h))

    @staticmethod
    def _get_prototype_category(tool_id: int) -> str:
        if 1228 <= tool_id <= 3227:
            return "nonce_optimization"
        elif 3228 <= tool_id <= 4227:
            return "consensus_stabilization"
        elif 4228 <= tool_id <= 5227:
            return "validation"
        return "pattern_analysis"

    # ------------------------------------------------------------------
    # FEM proxy
    # ------------------------------------------------------------------
    @staticmethod
    def finite_element_solution(a_k: float) -> float:
        def f(x): return float(a_k) * (x**2)
        res, _ = quad(f, 0, 1)
        return abs(float(res))

    # ------------------------------------------------------------------
    # Simple quantum walk (utility for hybrid methods)
    # ------------------------------------------------------------------
    @staticmethod
    def simple_quantum_walk(steps: int) -> np.ndarray:
        # Return a normalized probability vector over positions [-steps, steps]
        size = 2 * steps + 1
        # Use a symmetric distribution approximating a Hadamard walk footprint
        positions = np.arange(-steps, steps + 1, dtype=float)
        probs = np.exp(-(positions**2) / max(1, steps/2))
        probs /= probs.sum()
        return probs

    # ------------------------------------------------------------------
    # Crypto combo helpers (from crypto.py, patched)
    # ------------------------------------------------------------------
    @staticmethod
    def symplectic_kaprekar(x_in: int, p: float) -> int:
        H = 0.5 * (p**2) + math.sin(x_in)
        val = MathToolBox.kaprekar_operation(x_in) * math.exp(-H)
        return int(val) % (2**32)

    @staticmethod
    def ergodic_nonce_sequence(seed: int, n: int) -> List[float]:
        rng = np.random.default_rng(seed)
        mu = rng.random()
        seq = [float(seed)]
        for _ in range(n - 1):
            nxt = (seq[-1] + rng.normal(0, 1)) * mu % (2**32)
            seq.append(float(nxt))
        return seq

    @staticmethod
    def extreme_value_nonce_predictor(data: List[float]) -> float:
        loc, scale = gumbel_r.fit(data)
        return float(gumbel_r.rvs(loc=loc, scale=scale, size=1)[0])

    @staticmethod
    def copula_nonce_model(data_x: List[float], data_y: List[float], copula_type: str = 'gaussian') -> np.ndarray:
        if not COPULAS_AVAILABLE:
            raise ImportError("copulas package required")
        copula_map = {'gaussian': CopulaGaussian, 'clayton': CopulaClayton}
        copula_cls = copula_map.get(copula_type.lower(), CopulaGaussian)
        cop = copula_cls()
        data = np.column_stack((np.asarray(data_x), np.asarray(data_y)))
        cop.fit(data)
        return cop.sample(1)[0]

    @staticmethod
    def nimber_game_nonce(seed: int) -> int:
        # Simple nim-sum inspired transform
        # XOR fold chunks of the seed to produce a high-entropy 32-bit integer
        x = seed & 0xFFFFFFFF
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        return x

    @staticmethod
    def motivic_cohomology_nonce(p: float) -> int:
        # Interpret p as a parameter; create a bounded integer via smooth periodic map
        val = int(abs(1e6 * math.sin(p) * math.cos(p/2))) & 0xFFFFFFFF
        return val

    def master_nonce_predictor(self, seed: int, p: float) -> int:
        kap = self.symplectic_kaprekar(seed, p)
        erg_seq = self.ergodic_nonce_sequence(seed, 10)
        evt = self.extreme_value_nonce_predictor(erg_seq)
        cop = self.copula_nonce_model(erg_seq, np.random.permutation(erg_seq)) if COPULAS_AVAILABLE else np.array([0.0, 0.0])
        nim = self.nimber_game_nonce(seed)
        mot = self.motivic_cohomology_nonce(p)
        combined = float(kap) + float(evt) + float(cop[0] if isinstance(cop, np.ndarray) else 0.0) + float(nim) + float(mot)
        return int(combined) % (2**32)

    # ------------------------------------------------------------------
    # Optimization / hybrid nonce
    # ------------------------------------------------------------------
    def sha256_zkoraclenumerical_entropy_synergy(self, k: int, s: float, h: float, sigma: float, batch_size: int = 4096) -> np.ndarray:
        a_k = self.pi_fibonacci_enhanced(k)
        out = np.zeros(batch_size, dtype=float)
        for i in range(batch_size):
            p_adic = int(mp.nint(a_k)) % 9
            term = mp.log(1 + mp.fabs(mp.sin(a_k * s * h * sigma * mp.mpf('1e-20'))))
            out[i] = float(p_adic * mp.fabs(term) * mp.tanh(a_k * s * h * mp.mpf('1e-10')) % 9)
        return out

    def quantum_classical_hybrid_nonce(self, steps: int, x: int) -> int:
        probs = self.simple_quantum_walk(steps)
        classical_nonce = self.master_nonce_predictor(x, math.pi)
        return int(np.sum(probs) * classical_nonce) % (2**32)

# ----------------------------------------------------------------------------
# Module-level convenience exports
# ----------------------------------------------------------------------------
__all__ = [
    "MathToolBox",
    "Functor",
    "compose_functors",
]
