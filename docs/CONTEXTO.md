# CONTEXTO.md

Proyecto de simulación numérica para el modelo híbrido De Broglie–Bohm / SED.
Este documento es el contexto operativo del repo: qué hay, qué hicimos, qué
sigue. Sirve tanto como referencia personal como contexto para Claude Code
cuando trabajemos en este repo.

---

## 1. Identidad del proyecto

**Autor**: Santiago Bergamin (becario doctoral CONICET, Mendoza, Argentina)
**Director**: E. Bringa (experto en simulaciones y sistemas complejos)
**Línea de investigación**: fundamentos cuánticos — derivación dinámica
de la regla de Born a partir de mecánica de Bohm + electrodinámica
estocástica (SED).

**Pregunta de tesis**: ¿el campo electromagnético de punto cero (ZPF)
puede actuar como mecanismo físico que termalice ensambles de partículas
bohmianas hacia la distribución $|\psi|^2$?

**Estado actual**: Paper-1 listo para submission. Paper-2 Fases 1–3 completas
con auditoría metodológica (mayo 2026). Sweep canónico Np=10000 con corrección
Miller-Madow ya en `results/paper2_fase3/prod_Np10000/`. Fase 4 (manuscrito)
en curso: `papers/paper2/hybrid_AL_paper.tex` pendiente de actualización con
los números post-auditoría.

---

## 2. Resultado central del Paper-1 (lo que ya está cerrado)

Construimos un Lagrangiano híbrido BdB/SED con un único parámetro $\lambda$
y derivamos el **Teorema 1**: una ley de relajación
$\Gamma(\lambda) \propto \lambda^2 D \pi^2 / L^2$ hacia $|\psi|^2$,
bajo cuatro hipótesis explícitas.

**Las cuatro hipótesis del Teorema 1:**
1. Aproximación markoviana ($\tau_c \ll \tau_{dB}$)
2. Ausencia de microestructura inicial en $\rho$
3. $|\psi|^2$ sin nodos de medida positiva
4. **Balance detallado del ZPF respecto de $|\psi|^2$** ← *la nueva*

**Hallazgo:** (i) y (ii) y (iii) son las hipótesis clásicas de Valentini;
(iv) la identificamos nosotros. Las simulaciones de alta estadística
muestran que:

- En **doble rendija** (geometría abierta, $|\psi|^2$ suave), la hipótesis
  (iv) se aproxima sobre la marginal $\bar{H}_x$. Resultado:
  $R^2 = 0{,}99$, $C = 3{,}68 \pm 0{,}37$. **El Teorema 1 se confirma.**

- En **caja cerrada $[0,\pi]^2$** (Valentini–Westman, superposición
  multimodal, $|\psi|^2$ con fuerte estructura nodal), la hipótesis (iv)
  falla. Resultado: $C < 0$, disrupción de la relajación con
  significancia $> 5\sigma$ a partir de $\lambda \approx 0{,}1$.
  **El Teorema 1 no aplica.**

- En **autoestado real $\varphi_{11}$** (test diagnóstico con
  $v_{BM} \equiv 0$), el ZPF aislado **saca** del equilibrio una
  distribución inicial $\rho_0 = |\psi|^2$, confirmando que el campo de
  modos planos del vacío no preserva $|\psi|^2$ como estado estacionario.

**Interpretación física:** el acoplamiento mínimo
$-(\lambda e/c)\dot q \cdot A_{ZPF}$ produce difusión isotrópica que
termaliza hacia la **distribución uniforme**, no hacia $|\psi|^2$. Para
recuperar la regla de Born hace falta un término adicional: la **deriva
osmótica de Nelson** $v_{osm} = D \nabla \ln |\psi|^2$, que en el
Lagrangiano debería emerger del mecanismo de fluctuación–disipación
asociado a la **radiación de reacción de Abraham–Lorentz**.

---

## 3. Estructura del repo (post-reorganización 2026-05-21)

```
HibridLorentz/
├── README.md, .gitignore
│
├── src/                              # código de simulación
│   ├── paper2/                       # Fase 3 ALD (activo, post-auditoría)
│   │   ├── bohm_zpf_AL_box.py
│   │   ├── abraham_lorentz_classic.py
│   │   ├── sympy_AL_derivation.py
│   │   └── in.bohm_box
│   └── paper1/                       # legacy
│       ├── bohm_zpf_box.py, bohm_zpf_lammps.py
│       ├── calibrate_zpf.py, in.bohm_zpf
│
├── analysis/                         # analyze_AL, analyze_npconv, analyze_box, plot_publication
├── slurm/                            # scripts SLURM (paths actualizados a src/paper2/, results/paper2_fase3/)
│
├── papers/{paper2,paper1}/
├── refs/                             # boyer1975.pdf, valentini2005.pdf, boyer_1975_results.json
├── docs/                             # este archivo, README_results.md, contexto.txt
│
├── figures/
│   ├── paper2/   fig_AL_{D_scaling,hbar_curves,tau_comparison,Npconv,final_Np10k,threshold,stationarity_audit}.{pdf,png}
│   ├── paper1/   fig{1..4}_*.{pdf,png}
│   └── boyer/    boyer_*sweep.pdf, box_analysis_nr20.pdf
│
├── results/
│   ├── paper2_fase3/
│   │   ├── prod_Np5000/              # sweep original (Nr=50, sin Miller-Madow) — método ref.
│   │   ├── prod_Np10000/             # sweep canónico (Nr=50, Miller-Madow)
│   │   ├── npconv/Np{2500,5000,10000}/
│   │   ├── AL_box/, AL_box_nelson/, AL_stat/
│   └── paper1/
│       ├── box/, box_om3/, run_10k/
│
└── logs/                             # SLURM stdout/stderr (gitignored)
```

**Convenciones que aplicamos:**

- Unidades adimensionales: $\hbar = m = 1$
- Outputs JSON con schema fijo: `{lambda, n_particles, n_realizations, hbar_mean, hbar_std, hbar_all, D_ALD, D_ALD_std, D_ALD_per_real, hbar_bias_corrected, ...}`
- Aleatoriedad reproducible: semillas explícitas (`np.random.default_rng(seed)`)
- Backend matplotlib: `Agg` (corremos headless en Clementina)
- Paralelización: SLURM arrays con `%5` throttle (5 nodos máximo)

**Convenciones que aplicamos:**

- Unidades adimensionales: $\hbar = m = 1$
- Outputs JSON con schema fijo: `{lambda, n_particles, n_realizations,
  hbar_mean, hbar_std, hbar_all, ...}`
- Aleatoriedad reproducible: semillas explícitas (`np.random.seed(seed)`)
- Backend matplotlib: `Agg` (corremos headless en Clementina)
- Paralelización: SLURM arrays con `%4` throttle (4 nodos máximo)

---

## 4. Stack técnico

**Cluster**: Clementina (CONICET), login `snmgt01`, usuario `sbergamin`.
- Python 3.9.18, NumPy, SciPy, Matplotlib (Agg), argparse
- LAMMPS con módulo PYTHON (co-simulación)
- SLURM scheduler, partición `cpunode` por defecto
- Proxy obligatorio: `http://172.28.3.3:3128`
- Sin CuPy en las corridas actuales (CPU-only por simplicidad)

**Métodos numéricos:**

- Split-operator FFT para propagación de $\psi$ (estable, unitario)
- Euler de primer orden para trayectorias bohmianas (con ZPF como ruido
  aditivo)
- Coarse-graining $16 \times 16$ celdas para $\bar{H}$ de Valentini
- Bootstrap sobre realizaciones ZPF para CIs al 95%

---

## 5. Paper-2: Próximo paso — derivación de Abraham–Lorentz

### 5.1 Pregunta de investigación

Si agregamos al Lagrangiano el término de reacción radiativa de
Abraham–Lorentz–Dirac (ALD),
$$\mathcal{L}_{rad} \propto \tau_{rad}\,\dot q \cdot \dddot q,
\qquad \tau_{rad} = \frac{2 e^2}{3 m c^3},$$
¿emerge automáticamente la deriva osmótica de Nelson
$v_{osm} = D \nabla \ln |\psi|^2$ en el promedio sobre realizaciones del
ZPF?

**Si la respuesta es sí**, entonces:
- El Teorema 1 deja de ser condicional (hipótesis (iv) se cumple de
  oficio).
- La caja cerrada de Valentini–Westman debería relajarse con $C > 0$
  cuando incluimos ALD.
- Sería la **primera derivación física de la mecánica estocástica de
  Nelson a partir de SED real**, no postulada ad hoc.

### 5.2 Por qué esto es plausible (no es una apuesta a ciegas)

La idea no es nuestra — la propuso de la Peña & Cetto (lSED, "linear
SED") en *The Quantum Dice* (1996) y *The Emerging Quantum* (2015). Su
argumento físico es:

1. El ZPF empuja a la partícula con fluctuaciones aleatorias (fuente).
2. La partícula, al ser acelerada, irradia (reacción radiativa).
3. En estado estacionario, energía absorbida = energía irradiada
   (relación de fluctuación–disipación de Kubo).
4. Esta relación implica que la dinámica preserva una medida invariante
   específica — que dlP–C argumentan que es $|\psi|^2$.

Lo que falta en la literatura: una **demostración numérica** de que
esto efectivamente ocurre en una geometría no trivial. Eso es lo que
nosotros podemos aportar.

### 5.3 Riesgos teóricos conocidos

ALD tiene problemas matemáticos clásicos:

- **Soluciones runaway**: una partícula libre acelera exponencialmente
  hacia $\infty$. Se elude con la aproximación de **Landau–Lifshitz**
  que reemplaza $\dddot q \to (1/m) \dot F_{ext}$.
- **Preaceleración**: la partícula se acelera *antes* de que la fuerza
  externa actúe. Es no-causal pero a escalas $\sim \tau_{rad} \sim
  10^{-24}$ s, irrelevante para el rango temporal del modelo.
- **Decisión técnica:** usaremos Landau–Lifshitz desde el principio.

### 5.4 Plan de trabajo (4 fases, ~12 meses)

#### Fase 1 — Derivación teórica (mes 1–2) ✅ COMPLETA

**Entregable:** `sympy_AL_derivation.py` — derivación simbólica completa.

Tareas:
- [x] Agregar $\mathcal{L}_{rad}$ al Lagrangiano del Paper-1
- [x] Re-derivar las ecuaciones de Euler–Lagrange con SymPy
- [x] Aplicar aproximación Landau–Lifshitz: EoM-LL con fricción
      posición-dependiente $\gamma(q) = \tau_{rad} V''(q)/m$
- [x] Fokker–Planck de la EoM-LL: condición J=0 → $\rho_{stat} = |\psi|^2$
- [x] Relación-puente: $D_{eff} \cdot \tau_{rad} = \hbar/(2m\omega_0^2)$
- [x] Verificar $\langle x^2\rangle_{SymPy} = 1/(2\omega_0)$ [Boyer 1975 ✓]

**Resultados clave:**

1. La ALD/LL introduce fricción $\gamma(q) = \tau_{rad}V''(q)/m$ y ruido
   modificado $\xi_{eff} = f_{ZPF} + \tau_{rad}\dot{f}_{ZPF}$.
2. El FDT (T=0, ZPF) da $D_{eff}(x) = \hbar\omega_{local}(x)/(2m)$.
3. La condición $\rho_{stat} = |\psi|^2$ equivale a que $\psi$ satisfaga
   la ecuación de Schrödinger con $V_{ext}$ (auto-consistencia).
4. $v_{osm} = D\nabla\ln|\psi|^2$ emerge sin postulado adicional.

**Tensión identificada:** $D_{FDT} = \hbar\omega_0/(2m)$ vs. $D_{Nelson} = \hbar/(2m)$
— la resolución requiere el tratamiento de modos completo (Fase 2).

**Producto:** Teorema 1 sin hipótesis (iv), bajo la condición de que
$\psi$ sea solución de Schrödinger (auto-consistencia SED).

#### Fase 2 — Validación numérica clásica (mes 3–4) ✅ COMPLETA

**Entregable:** `abraham_lorentz_classic.py` — Euler simpléctico + FDT cuántico.

- [x] Euler simpléctico (det M = 1–γdt < 1, estable para dt < 2/ω₀)
- [x] Boyer (1975) reproducido: ω₀ ∈ {0.5,1.0,1.5,2.0,3.0} → ⟨x²⟩=1/(2ω₀), todos ≤1.5σ
- [x] Universalidad: τ_rad ∈ {0.005,...,0.1} → ⟨x²⟩ = cte. independiente de τ_rad
- [x] Equipartición: ⟨v²⟩/(ω₀²⟨x²⟩) = 1.000±0.003 en todos los casos

**Bug clave resuelto:** Euler explícito diverge (dt > τ_rad). Euler simpléctico resuelve.

**Tensión D_FDT vs D_Nelson:** D_FDT = ω₀/2 (depende de ω₀ para el OA). La universalidad
D = ℏ/2m requiere integración sobre todos los modos ZPF → probado en Fase 3.

#### Fase 3 — Simulación híbrida con ALD (mes 5–8) ✅ COMPLETA POST-AUDITORÍA

**Entregable:** `src/paper2/bohm_zpf_AL_box.py` + sweep canónico Np=10000 + auditoría metodológica.

**Pipeline final del driver:**
- ZPFField con `field_at()` y `field_dot_at()` (LL correction).
- D_ALD analítico `λ²·Σ A_k²·τ_c·Δt/4`, recolectado y promediado por realización.
- `vx += lam·(A + τ_rad·Ȧ)` + osmotic drift `v_osm = D·∇ln|ψ|²` always-on.
- **Miller–Madow correction** sobre $\bar H$: $(K_{eff}-1)/(2 N_p)$ (clave: removió un sesgo de ~13% en τ que disfrazaba la convergencia).
- `V_CLIP = 200` sobre v_x, v_y antes del Euler (seguridad cerca de nodos).
- JSON gana `hbar_bias_corrected`, `D_ALD_std`, `D_ALD_per_real`.

**Auditoría completa (May 2026), 5 chequeos:**

1. **Convergencia en Np** (sweep auxiliar Np ∈ {2500, 5000, 10000} × λ ∈ {0, 0.10}, Nr=50):
   detectó τ no convergido bajo el estimador histograma → identificó el sesgo Miller-Madow como artefacto dominante.
   Post-corrección: τ es plano en Np ✓ (escala 1/Np desaparece, R² 0.999).

2. **Sweep canónico Np=10000** (SLURM 1218153, 10 tareas, exit 0, Nr=50, τ_rad=0.01, ω_max=3.0):

   | λ      | τ_eff ± σ      | Γ = 1/τ | ΔΓ        |
   |--------|----------------|---------|-----------|
   | 0.000  | 4.929 ± 0.014  | 0.2029  | 0         |
   | 0.005  | 4.898 ± 0.015  | 0.2042  | +0.0013   |
   | 0.010  | 4.899 ± 0.016  | 0.2041  | +0.0012   |
   | 0.020  | 4.900 ± 0.019  | 0.2041  | +0.0012   |
   | 0.030  | 4.899 ± 0.023  | 0.2041  | +0.0012   |
   | 0.050  | 4.963 ± 0.026  | 0.2015  | −0.0014   |
   | 0.070  | 5.065 ± 0.036  | 0.1974  | −0.0055   |
   | 0.100  | 5.290 ± 0.053  | 0.1890  | −0.0139   |
   | 0.150  | 5.809 ± 0.086  | 0.1721  | −0.0308   |
   | 0.200  | 6.518 ± 0.116  | 0.1534  | −0.0495   |

3. **Modelo funcional — threshold gana por AIC** (vs cuártico puro y mixto λ²+λ⁴):
   $$\Delta\Gamma(\lambda) = -C\,(\lambda^2 - \lambda_c^2)\,\theta(\lambda - \lambda_c)$$
   - $C = 1.32 \pm 0.07$, $\lambda_c = 0.030 \pm 0.012$, χ²/dof = 1.19
   - Fit naive `C·λ²` desde origen sobreestima la pendiente ~17% y tiene residual de +3.8σ a λ=0.20; el threshold queda en <1σ.

4. **Boyer 1975 benchmark** (`abraham_lorentz_classic.py`, `boyer_1975_results.json`):
   con τ_rad=0.01 (el valor de Fase 3), `<x²>_stat = 0.5025` vs predicción exacta `0.500` → **0.5% off**. Friction ALD implementado correctamente ✓.

5. **Stationarity audit** (run_stationary_AL, λ=0.10, código corregido):
   - born_ic: H̄ ≈ 0.08 ± 0.02 (estable, ruido O(1/Np²)) ✓
   - uniform_ic: H̄ ≈ 1.16 → 1.23 en t=4π (sin relajación). Esperado: τ_drift = ε/(2D) ≈ 1200 ≫ t_final.
   - **Interpretación clave**: τ_BdB ≈ 5 (box, mixing de 16 modos) ≪ τ_ZPF→Born ≈ 1200 (φ₁₁ solo). Los dos mecanismos están separados por dos órdenes de magnitud → ZPF NO puede impulsar la relajación rápida; sólo la disrumpe.

**Scripts:** `src/paper2/bohm_zpf_AL_box.py`, `analysis/{analyze_AL,analyze_npconv}.py`, `slurm/slurm_AL_sweep_Np10k.sh`.

#### Fase 4 — Paper-2 (mes 9–12) — EN CURSO

**Archivo:** `papers/paper2/hybrid_AL_paper.tex` (esqueleto LaTeX RevTeX4-2; pendiente actualizar con números canónicos post-auditoría).
**Objetivo de revista**: *Physical Review Letters* (4 páginas) o *Foundations of Physics*.

Estructura actual del skeleton:
- Sec. I: Introduction (dBB, Valentini, ALD motivation)
- Sec. II: ALD+FDT → Nelson osmotic (derivación, tensión D_ALD vs D_Nelson)
- Sec. III: Boyer benchmark (validado: 0.5% off de la predicción exacta)
- Sec. IV: Valentini box (stationarity test + sweep canónico Np=10000)
- Sec. V: Discussion (threshold λ_c, separación de escalas τ_BdB ≪ τ_ZPF, outlook)
- Sec. VI: Conclusions

**Resultado central del Paper-2 (post-auditoría — DEFINITIVO):**

El sweep canónico Np=10000 con Miller-Madow muestra que **la radiación de reacción ALD NO invierte la disrupción del ZPF**. La conclusión preliminar `C = +0.42` (basada en Nr=5 y H̄ sin bias-correction) era artefacto del estimador histograma. El resultado real es:

$$\boxed{\;\Delta\Gamma(\lambda) = -C\,(\lambda^2 - \lambda_c^2)\,\theta(\lambda-\lambda_c),\quad C = 1.32 \pm 0.07,\quad \lambda_c = 0.030 \pm 0.012\;}$$

Es decir: ZPF + ALD **disrumpe** la relajación bohmiana en la caja cerrada, con un umbral cinético λ_c por debajo del cual el efecto es indetectable. Esto contradice la intuición de la Peña-Cetto, y la razón estructural está cuantificada en la **separación de escalas τ_BdB ≈ 5 vs τ_ZPF→Born ≈ 1200**: la deriva osmótica que ALD provee es 200× demasiado lenta para competir con la dinámica BdB intrínseca, así que el efecto neto del ZPF es perturbar la relajación rápida en lugar de impulsar una nueva.

---

## 6. Archivos nuevos previstos

Cuando arranquemos el Paper-2, estos serán los nuevos archivos. **No
crear todavía.** Lista de referencia para cuando lleguemos a cada fase:

```
proyecto/
├── 📐 Paper-2: derivación teórica (Fase 1)
│   ├── sympy_AL_derivation.py      # cálculo simbólico Euler–Lagrange con ALD
│   ├── derivation_AL.tex           # documento de la derivación
│   └── derivation_AL.pdf
│
├── 🧪 Paper-2: validación clásica (Fase 2)
│   ├── abraham_lorentz_classic.py  # oscilador armónico SED puro + ALD
│   ├── test_boyer_1975.py          # reproducción del benchmark
│   └── boyer_1975_results.json
│
├── 🚀 Paper-2: simulación híbrida (Fase 3)
│   ├── bohm_zpf_AL_box.py          # BdB + ZPF + ALD en caja cerrada
│   ├── in.bohm_AL_box              # input LAMMPS extendido
│   ├── analyze_AL.py               # análisis específico Paper-2
│   └── slurm_AL_sweep.sh           # array job para barrido de λ
│
├── 📦 Resultados Paper-2 (Fase 3)
│   └── AL_results_lam0_*.json
│
└── 📄 Paper-2: manuscrito (Fase 4)
    └── hybrid_AL_paper.tex
```

---

## 7. Lecciones del Paper-1 (no repetir errores)

Bugs que nos costaron meses; documentados para no volver a caer:

1. **Inicialización inconsistente entre fases.** El KS de 62% inicial era
   artefacto: distintas `SimConfig` por defecto entre fase preparación
   y fase corrida. → **Lección**: un único punto de configuración por
   experimento, no defaults distribuidos.

2. **`run_simulation()` no aceptaba `q_scale`**: el test de
   Q-replacement caía en un fallback de post-procesamiento que
   interpolaba trayectorias en lugar de resolver ecuaciones reales. →
   **Lección**: cualquier parámetro físico que el test varíe debe ser
   argumento explícito de la función de simulación, no inferido por
   ramas silenciosas.

3. **Madelung directo es inestable**. Integrar $\rho$, $S$ con RK4 falla
   cuando $\rho \to 0$ cerca de nodos. → **Solución actual**:
   split-operator FFT sobre $\psi$. Mantener.

4. **Wrap periódico en $\bar{H}_{2D}$**: la métrica 2D tenía un
   artefacto cuando el ensamble cruzaba el borde periódico antes que la
   función de onda. → **Solución**: marginal $\bar{H}_x$ (ya
   implementada en `analyze_relaxation.py`).

5. **Condición inicial cerca de equilibrio = no se ve nada.** Si
   $\rho_0 \approx |\psi|^2$ el sistema ya está termalizado y la métrica
   $D_{KS}$ crece monótonamente con $\lambda$ (efecto ZPF puro de
   disrupción). → **Lección**: condiciones iniciales fuera de
   equilibrio son obligatorias para ver la relajación.

---

## 8. Notas operativas (para Claude Code y para mí)

- **Idioma**: código y commits en inglés, comentarios físicos en
  español, conversación en español.
- **Output formats**: `.tex` o `.md` preferidos; PDF solo cuando se
  pide.
- **Estilo de respuesta**: directo, sin preámbulos, sin
  expansión no pedida.
- **SLURM scripts**: ASCII puro, line endings Unix,
  `export MPLBACKEND=Agg` siempre, proxy con `http://` explícito.
- **Validación estadística**: bootstrap sobre realizaciones ZPF,
  CIs al 95%, $z$-scores combinados respecto del baseline $\lambda = 0$.
- **Antes de declarar resultados**: convergencia en $N_r$ (mínimo 20
  realizaciones para significancia confiable).

---

## 9. Referencias clave (para tener a mano)

**Paper-1 (nuestro)**: Bergamin & Bringa (2026), *Lagrangian
formulation of a hybrid De Broglie–Bohm / stochastic electrodynamics
model* — `hybrid_paper_unified.tex`.

**Para Paper-2 (Abraham–Lorentz + lSED):**

- de la Peña, Cetto & Valdés-Hernández (2015), *The Emerging Quantum:
  The Physics Behind Quantum Mechanics*, Springer. **Capítulo 4** es la
  derivación lSED de la regla de Born.
- Boyer (1975), Phys. Rev. D 11, 790 — el benchmark del oscilador
  armónico SED.
- Rohrlich (2007), *Classical Charged Particles*, World Scientific —
  tratamiento moderno de Abraham–Lorentz, incluye Landau–Lifshitz.
- Nelson (1966), Phys. Rev. 150, 1079 — la mecánica estocástica que
  queremos derivar como caso límite.

**Para entender el cuello de botella numérico:**

- Valentini & Westman (2005), Proc. R. Soc. A 461 — la geometría
  canónica que usamos en `bohm_zpf_box.py`.
- Towler, Russell & Valentini (2012), Proc. R. Soc. A 468 — estimaciones
  de $\tau_V$ con coarse-graining; útil para sanity-check de
  $\tau_{eff}(\lambda = 0)$.

---

## 10. Evaluación de las simulaciones — qué salió bien y qué salió mal

### ✅ Lo que salió bien (resultados sólidos para el manuscrito)

| Chequeo | Resultado | Implicación |
|---|---|---|
| **Boyer 1975 (Fase 2)** | `<x²>_stat = 0.5025` vs predicción `0.500` con τ_rad=0.01 (0.5% off) | El friction ALD está implementado correctamente. Test pasado limpio. |
| **Convergencia Np** | Tras Miller-Madow, τ es plano en Np ∈ {2500, 5000, 10000} (1/Np scaling con R²=0.999 antes, plano después) | Datos canónicos a Np=10k son asintóticamente convergidos a <1%. |
| **Stationarity born_ic** | H̄ ≈ 0.08 ± 0.02 estable (residuo O(1/Np²)) | Código preserva el equilibrio Born como debe. |
| **Stationarity uniform_ic** | No relaja en t=4π, consistente con τ_drift ≈ 1200 calculado analíticamente | Comportamiento esperado; valida la estimación FDT. |
| **Fit threshold** | C=1.32±0.07, λ_c=0.030±0.012, χ²/dof=1.19, AIC=13.5 (vs naive 20.0+) | Modelo funcional robusto y físicamente interpretable. |
| **Significancia estadística** | Δτ(λ=0.20) = +1.59 con σ=0.12 → 13σ del baseline | Efecto físico no atribuible a ruido. |
| **Acuerdo Np-extrapolación** | τ₀ canónico 4.929 ± 0.014 vs extrapolación 1/Np 4.94 | <1% de match. Pipeline cerrado. |

### ⚠️ Lo que salió mal o requiere matiz crítico

| Hallazgo | Severidad | Acción |
|---|---|---|
| **El "C = +0.42" de la corrida preliminar (Nr=5) era artefacto del estimador H̄** sin Miller-Madow + muy poca estadística | 🔴 Crítico | Resuelto: descartado, reemplazado por C = −1.32 (con la convención del paper) en el sweep canónico. |
| **El programa lSED de la Peña-Cetto NO se valida**: agregar ALD no genera la relajación Born-rule que ellos predicen — sigue siendo disruptiva, igual que el Paper-1 sin ALD | 🟡 Resultado físico | Resignificar el manuscrito: pasa de "ALD activa relajación" a "ALD no es suficiente; la disrupción persiste". Igualmente publicable, pero con narrativa opuesta. |
| **D_ALD ≪ D_Nelson** (ratio ~10⁻⁵ en régimen perturbativo): la deriva osmótica que ALD provee es 200× más lenta que τ_BdB | 🟡 Limitación conocida | Mencionar como motivación para Fase 4: tratar el continuo ZPF (no 32 modos discretos) puede cerrar la brecha. |
| **El régimen λ ≤ λ_c es indetectable** dentro del ruido: el efecto físico solo emerge a partir de λ ≈ 0.05 | 🟢 Aceptable | Threshold λ_c queda como predicción cuantitativa del modelo, no como debilidad. |
| **Sesgo residual O(1/N²)** del estimador: a λ chico hay un offset +0.0013 sistemático en ΔΓ que no se cancela con Miller-Madow | 🟢 Menor | Discutir en apéndice; desaparece en el fit threshold (intercept forzado a 0). |
| **Comparación cruzada Paper-1 vs Paper-2** tiene un offset metodológico ≈ 1.1 en τ₀ (distinto code path, distinto Nr) | 🟢 Cosmético | Tratar como artefacto de pipeline en el texto; el resultado se sostiene **dentro** de cada sweep. |

### 🎯 Veredicto general

**Las simulaciones son metodológicamente sólidas** tras la auditoría:
1. Friction ALD: validado independientemente (Boyer).
2. Estimador H̄: corregido (Miller-Madow), convergencia confirmada.
3. Sweep canónico: 10/10 tareas exit 0, fits robustos, errores cuantificados.

**El resultado físico no es el que esperábamos** (la Peña-Cetto), pero **es un hallazgo legítimo y publicable**: la radiación de reacción ALD, aplicada como Landau-Lifshitz a la corriente bohmiana, no genera la deriva osmótica de Nelson en magnitud suficiente para termalizar a |ψ|² en la caja cerrada. La conclusión cuantitativa robusta es un coeficiente de disrupción C = 1.32 con threshold λ_c = 0.030.

**El paper tiene una historia clara**: ZPF puro disrumpe → agregar ALD no salva el cuadro → la separación de escalas τ_BdB ≪ τ_ZPF explica por qué → trabajo futuro: ZPF continuo / sistemas con BdB más lento donde D_ALD podría dominar.

---

## 11. Estado actual a una línea

Paper-1 listo (sin cambios). Paper-2 Fases 1–3 completas con auditoría;
resultado canónico C = 1.32 ± 0.07, λ_c = 0.030 ± 0.012 (threshold).
Próximo: actualizar `hybrid_AL_paper.tex` con números post-auditoría, push de los 2 commits de reorg, decidir revista objetivo.---
name: ahorro-tokens
description: Reglas para reducir tokens en Claude Code
---

#
meta:
  title: "Reglas para Claude Code — Ahorra Tokens"
  instruction: "Copia este contenido en `CLAUDE.md` en la raiz de tu proyecto o en `~/.claude/CLAUDE.md` para que aplique a todos tus proyectos."
rules:
  - number: 1
    title: "No programar sin contexto"
    description: |
      ANTES de escribir codigo: lee los archivos relevantes, revisa git log, entiende la arquitectura.
      Si no tienes contexto suficiente, pregunta. No asumas.
  - number: 2
    title: "Respuestas cortas"
    description: |
      Responde en 1-3 oraciones. Sin preambulos, sin resumen final.
      No repitas lo que el usuario dijo. No expliques lo obvio.
      Codigo habla por si mismo: no narres cada linea que escribes.
  - number: 3
    title: "No reescribir archivos completos"
    description: |
      Usa Edit (reemplazo parcial), NUNCA Write para archivos existentes salvo que el cambio sea >80% del archivo.
      Cambia solo lo necesario. No "limpies" codigo alrededor del cambio.
  - number: 4
    title: "No releer archivos ya leidos"
    description: |
      Si ya leiste un archivo en esta conversacion, no lo vuelvas a leer salvo que haya cambiado.
      Toma notas mentales de lo importante en tu primera lectura.
  - number: 5
    title: "Validar antes de declarar hecho"
    description: |
      Despues de un cambio: compila, corre tests, o verifica que funciona.
      Nunca digas "listo" sin evidencia de que funciona.
  - number: 6
    title: "Cero charla aduladora"
    description: |
      No digas "Excelente pregunta", "Gran idea", "Perfecto", etc.
      No halagues al usuario. Ve directo al trabajo.
  - number: 7
    title: "Soluciones simples"
    description: |
      Implementa lo minimo que resuelve el problema. Nada mas.
      No agregues abstracciones, helpers, tipos, validaciones, ni features que no se pidieron.
      3 lineas repetidas > 1 abstraccion prematura.
  - number: 8
    title: "No pelear con el usuario"
    description: |
      Si el usuario dice "hazlo asi", hazlo asi. No debatas salvo riesgo real de seguridad o perdida de datos.
      Si discrepas, menciona tu concern en 1 oracion y procede con lo que pidio.
  - number: 9
    title: "Leer solo lo necesario"
    description: |
      No leas archivos completos si solo necesitas una seccion. Usa offset y limit.
      Si sabes la ruta exacta, usa Read directo. No hagas Glob + Grep + Read cuando Read basta.
  - number: 10
    title: "No narrar el plan antes de ejecutar"
    description: |
      No digas "Voy a leer el archivo, luego modificar la funcion, luego compilar...". Solo hazlo.
      El usuario ve tus tool calls. No necesita un preview en texto.
  - number: 11
    title: "Paralelizar tool calls"
    description: |
      Si necesitas leer 3 archivos independientes, lee los 3 en un solo mensaje, no uno por uno.
      Menos roundtrips = menos tokens de contexto acumulado.
  - number: 12
    title: "No duplicar codigo en la respuesta"
    description: |
      Si ya editaste un archivo, no copies el resultado en tu respuesta. El usuario lo ve en el diff.
      Si creaste un archivo, no lo muestres entero en texto tambien.
  - number: 13
    title: "No usar Agent cuando Grep/Read basta"
    description: |
      Agent duplica todo el contexto en un subproceso. Solo usalo para busquedas amplias o tareas complejas.
      Para buscar una funcion o archivo especifico, usa Grep o Glob directo.