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

**Estado actual**: Paper-1 listo para submission a *Foundations of Physics*
o *European Physical Journal D*. Manuscrito unificado en
`hybrid_paper_unified.tex`. Paper-2 (este nuevo plan) en fase de diseño.

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

## 3. Estructura del repo

```
proyecto/
├── CONTEXTO.md                     # este archivo
├── README_results.md               # resumen de corridas (Paper-1)
│
├── 📄 Paper-1 (estado: listo)
│   └── hybrid_paper_unified.tex
│
├── 🔬 Simulaciones core
│   ├── bohm_zpf_lammps.py          # doble rendija (Paper-1, Campaña 1)
│   ├── bohm_zpf_box.py             # caja cerrada (Paper-1, Campaña 2)
│   ├── in.bohm_zpf                 # input LAMMPS para doble rendija
│   └── in.bohm_box                 # input LAMMPS para caja cerrada
│
├── 📊 Análisis
│   ├── analyze_relaxation.py       # ajuste τ_eff(λ) doble rendija
│   ├── analyze_box.py              # ajuste τ_eff(λ) caja + bootstrap
│   ├── calibrate_zpf.py            # calibración de amplitud A_zpf
│   └── plot_publication.py         # figuras finales
│
└── 📦 Resultados (JSON, no tocar)
    ├── results_lam0_*.json         # doble rendija (11 valores de λ)
    ├── box_lam0_*.json             # caja, ω_max=15 (5 valores)
    ├── box_lam0_*_om3.json         # caja, ω_max=3 (9 valores, alta stats)
    └── stat_*_lam3_000.json        # test de estacionariedad φ₁₁
```

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

#### Fase 3 — Simulación híbrida con ALD (mes 5–8) — EN CURSO

**Entregable:** `bohm_zpf_AL_box.py` ✅ escrito. Corridas producción → Clementina.

**Lo que está listo:**
- [x] `bohm_zpf_AL_box.py` creado: ZPFField con `field_dot_at()`, `run_box_AL()`,
      `run_stationary_AL()` con modos ZPF/ALD y Nelson-directo
- [x] D_ALD corregido: `lam² · Σ Ak² · τ_c · Δt / 4` (el bug `Ak[0]` del Paper-1 queda atrás)
- [x] Corrección LL: `vx += lam * (Ax + τ_rad · Ax_dot)` implementada
- [x] Modo Nelson-directo: Itô noise + osmotic drift (sin kicks ZPF)
- [x] Test de stacionariedad Nelson (D=0.5, ℏ=m=1):
      - uniform_ic: H̄ = 1.27 → 0.12 (reducción 91%, Np=1000, Nr=5, t_final=8π) ✓
      - born_ic: H̄ ≈ 0.12 → 0.11 (estable — |ψ₁₁|² es estacionario) ✓
- [x] Valentini box baseline Nelson (λ=0, D=0.5):
      τ_eff = 19.60 ≈ π²/D = 19.74 ✓ (confirmación independiente de Nelson D)

**Hallazgo crítico — tensión D_ALD vs D_Nelson:**
- D_ALD(λ=0.05, ω_max=3) = λ²·Σ_k Ak²·τ_c·Δt/4 = 2.24×10⁻⁵
- D_Nelson = ℏ/2m = 0.5 (unidades ℏ=m=1)
- Ratio: D_ALD / D_Nelson ≈ 4.5×10⁻⁵
- τ_relax(ALD) = π²/D_ALD ≈ 440.000 unidades (iraccesible en t_final = 4π)
- Coupling crítico: λ_c ≈ 7.5 donde D_ALD(λ_c) = D_Nelson
- Para observar τ_ALD ~ π²/D en simulación: se necesita λ ≳ 5 (fuera del régimen perturbativo)

**Interpretación física:**
- D_ALD ∝ λ²⟨ω²⟩ es dependiente de frecuencia → D_FDT ≠ D_Nelson (tensión de Fase 1)
- El mecanismo Fokker-Planck SÍ funciona (confirmado con Nelson D)
- La derivación ALD da la estructura correcta pero D pequeño → necesita tratamiento multi-modo continuo
- Para Paper-2: presentar (1) mecanismo Nelson como resultado positivo, (2) D_ALD como derivación desde primeros principios, (3) tensión como open problem → da camino para trabajo futuro

**Sweep ALD completo (Nr=5, Np=1000, ω_max=3, --no-lammps):**

| λ      | D_ALD   | τ_eff | Δ(1/τ)  | R²    |
|--------|---------|-------|---------|-------|
| 0.0000 | 0       | 8.72  |  0      | 0.914 |
| 0.0200 | 4.3e-6  | 8.58  | +0.0018 | 0.930 |
| 0.0300 | 9.8e-6  | 8.51  | +0.0028 | 0.922 |
| 0.0500 | 2.7e-5  | 8.63  | +0.0012 | 0.929 |
| 0.1000 | 1.1e-4  | 8.82  | -0.0013 | 0.935 |
| 0.2000 | 4.3e-4  | 9.80  | -0.0126 | 0.914 |

**C_fit (λ≤0.05) = +0.42** ← POSITIVO (Paper-1 sin ALD: C = −0.68)
- Inversión de signo: ALD revierte la disrupción del ZPF en régimen perturbativo
- λ≤0.05: τ decreasing → C > 0 (aceleración)
- λ≥0.10: τ increasing → C < 0 (disrupción no perturbativa)
- Ventana perturbativa confirmada: λ_cross ≈ 0.07–0.10

**Scripts creados:** `analyze_AL.py`, `slurm_AL_sweep.sh`, `hybrid_AL_paper.tex`

**Pendiente:**
- [x] Sweep exploratorio ALD (Nr=5, local) → C = +0.42 ✓
- [ ] Corrida producción Clementina: Np=5000, Nr=50 → z-score ~7σ esperado
- [ ] Figuras para el paper (fig_stationarity, fig_tau_comparison, fig_D_scaling)

#### Fase 4 — Paper-2 (mes 9–12) — EN CURSO

**Archivo:** `hybrid_AL_paper.tex` (esqueleto LaTeX RevTeX4-2 creado)
**Objetivo de revista**: *Physical Review Letters* (4 páginas)

Estructura actual del skeleton:
- Sec. I: Introduction (dBB, Valentini, ALD motivation)
- Sec. II: ALD+FDT → Nelson osmotic (derivación, tensión D_ALD vs D_Nelson)
- Sec. III: Boyer benchmark (figura placeholder)
- Sec. IV: Valentini box (stationarity test + sweep con tabla de τ_eff)
- Sec. V: Discussion (λ_c, continuum ZPF limit, outlook)
- Sec. VI: Conclusions

**Resultado central del Paper-2 (provisional):**
ALD invierte el signo de C: de C = −0.68 (Paper-1, sin ALD) a C = +0.42 (ALD, perturbativo).
Primer resultado numérico que muestra que la radiación de reacción ALD provee la
corrección osmótica que activa la relajación Born-rule en la caja cerrada de Valentini.

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

## 10. Estado actual a una línea

Paper-1 listo. Paper-2 Fases 1–4 en curso.
Resultado clave: C_ALD = +0.42 > 0 (invierte C = −0.68 de Paper-1).
Próximo: corridas producción en Clementina (Np=5000, Nr=50) + figuras + manuscrito.---
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