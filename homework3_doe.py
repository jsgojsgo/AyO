"""
Tarea 3 - Ajuste del parámetro de temperatura (Recocido Simulado + 2-opt, TSP) con DOE
OPCIÓN A: diseño factorial completo 3 x 3 (temperatura x tamaño), 10 réplicas -> 90 corridas.

Uso:
    python tarea3_doe.py

Requiere: numpy, pandas, scipy, matplotlib, statsmodels
    pip install numpy pandas scipy matplotlib statsmodels

Salidas (carpeta Downloads / Descargas del usuario, prefijo t3_):
    gráficas PNG: efectos principales, interacción, bootstrap, cajas

"""
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")                       # guarda a archivos (no necesita ventana)
import matplotlib.pyplot as plt

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    HAY_STATSMODELS = True
except ImportError:                          # solo como respaldo; lo normal es tener statsmodels
    HAY_STATSMODELS = False

# =============================================================================
# [REQUISITO: la semilla de la matrícula. r se genera UNA sola vez. Después de estas dos líneas el script nunca vuelve
#  a llamar a random.random() global: todo lo demás usa generadores propios (random.Random(...)
#  o numpy default_rng), así que r no se regenera ni se altera.]
# =============================================================================
MATRICULA = 2115027
random.seed(MATRICULA)
r = random.random()

# =============================================================================
# PARÁMETROS DEL EXPERIMENTO
# =============================================================================
ALFA = 0.05                                  # [REQUISITO: ANOVA con alfa = 0.05 (IC 95%)]
TAMANOS = [5, 10, 15]                        # [REQUISITO: bloque = tamaño n en {5, 10, 15}]
REPLICAS = 10                                # [REQUISITO: 10 réplicas por punto de diseño]

# [REQUISITO: 10 semillas DISTINTAS entre sí, y el MISMO conjunto fijo en cada nivel de
#  temperatura (Common Random Numbers). Cada corrida arma su generador con una de estas semillas.]
SEMILLAS = [MATRICULA + k for k in range(1, REPLICAS + 1)]

# [REQUISITO: rango de temperatura propio [T_min, T_max]. Las ciudades viven en [0,100]^2,
#  así que las aristas típicas miden entre ~10 y ~60 y un movimiento 2-opt empeora el tour en
#  decenas de unidades. T_max = 50 acepta casi cualquier movimiento (exploración total) y
#  T_min = 0.1 prácticamente solo acepta mejoras (casi hill climbing).]
T_MIN, T_MAX = 0.1, 50.0

# [REQUISITO: Opción A, 3 niveles como FRACCIÓN del rango, calculados con r:
#  bajo = 0.3*r, medio = 0.3*r + 0.3, al    to = 0.3*r + 0.6]
FRACCIONES = {"bajo": 0.3 * r, "medio": 0.3 * r + 0.3, "alto": 0.3 * r + 0.6}
# [REQUISITO: conversión T_nivel = T_min + fraccion * (T_max - T_min)]
TEMPERATURAS = {nivel: T_MIN + f * (T_MAX - T_MIN) for nivel, f in FRACCIONES.items()}
NIVELES = list(TEMPERATURAS)                 # ["bajo", "medio", "alto"]

T_FINAL = 0.01                               # el enfriamiento exponencial va de T0 a T_FINAL
N_ITER = 100                                 # iteraciones por corrida (justificado con el piloto)
N_BOOT = 10_000                              # remuestras bootstrap
SEMILLA_BOOT = MATRICULA                     # generador numpy propio para el bootstrap

CARPETA = Path.home() / "Downloads"
CARPETA.mkdir(parents=True, exist_ok=True)


def guardar_fig(fig, nombre):
    ruta = CARPETA / nombre
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  [gráfica guardada] {ruta}")


# =============================================================================
# TSP: instancias, 2-opt y Recocido Simulado (obtenido de la Tarea 2)
# =============================================================================
def crear_instancia(n):
    """Una sola instancia FIJA por tamaño (misma semilla siempre; no cambia entre temperaturas)."""
    rng = random.Random(MATRICULA * 1000 + n)
    pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
    return [[math.dist(p, q) for q in pts] for p in pts]


def longitud_tour(D, tour):
    return sum(D[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def delta_2opt(D, t, i, k):
    """Cambio de longitud al invertir t[i..k] (se reemplazan las aristas (i-1,i) y (k,k+1))."""
    a, b, c, d = t[i - 1], t[i], t[k], t[(k + 1) % len(t)]
    return D[a][c] + D[b][d] - D[a][b] - D[c][d]


def recocido_simulado(D, T0, n_iter, semilla, T_final=T_FINAL):
    """
    Recocido simulado con 2-opt (Tarea 2, Ej. 4): solución inicial aleatoria, enfriamiento
    exponencial de T0 a T_final. Regresa el MEJOR costo (longitud de tour) hallado en la corrida.

    [REQUISITO: Common Random Numbers] En cada iteración se sortean SIEMPRE el par (i, k)
    y el número u de la prueba de aceptación, aunque no se necesite. Así, con la misma semilla,
    las propuestas de movimiento son idénticas en los tres niveles de temperatura y lo único que
    cambia entre ellos es la decisión de aceptar o no.
    """
    rng = random.Random(semilla)             # generador propio: no toca el estado global de r
    n = len(D)
    tour = rng.sample(range(n), n)
    actual = longitud_tour(D, tour)
    mejor, mejor_tour = actual, tour[:]
    for paso in range(n_iter):
        T = T0 * (T_final / T0) ** (paso / n_iter)
        i, k = sorted(rng.sample(range(1, n), 2))
        u = rng.random()
        delta = delta_2opt(D, tour, i, k)
        if delta < 0 or u < math.exp(-delta / T):
            tour[i:k + 1] = tour[i:k + 1][::-1]
            actual += delta
            if actual < mejor:
                mejor, mejor_tour = actual, tour[:]
    # se recalcula la longitud exacta del mejor tour (evita errores de punto flotante acumulados)
    return round(longitud_tour(D, mejor_tour), 6)


def optimo_held_karp(D):
    """Longitud del tour óptimo exacto (programación dinámica, O(n^2 2^n)); solo para referencia."""
    n = len(D)
    inf = float("inf")
    completo = 1 << (n - 1)
    dp = [[inf] * (n - 1) for _ in range(completo)]
    for j in range(n - 1):
        dp[1 << j][j] = D[0][j + 1]
    for mascara in range(1, completo):
        for j in range(n - 1):
            if not (mascara >> j) & 1 or dp[mascara][j] == inf:
                continue
            for k in range(n - 1):
                if (mascara >> k) & 1:
                    continue
                nueva = mascara | (1 << k)
                v = dp[mascara][j] + D[j + 1][k + 1]
                if v < dp[nueva][k]:
                    dp[nueva][k] = v
    return min(dp[completo - 1][j] + D[j + 1][0] for j in range(n - 1))


# =============================================================================
# ESTUDIO PILOTO (reconocimiento barato, con semillas distintas a las del experimento)
# =============================================================================
def estudio_piloto(instancias, optimos):
    """
    Sirve para fijar el presupuesto de iteraciones común N_ITER. Con demasiadas iteraciones todo
    algoritmo llega al óptimo en n = 5 y 10 (efecto de piso) y no se distingue ningún efecto.
    """
    presupuestos = [25, 50, 100, 200, 400, 800]
    semillas_piloto = [900_001 + k for k in range(REPLICAS)]     # distintas a SEMILLAS
    filas = []
    for N in presupuestos:
        for n in TAMANOS:
            brechas = []
            for nivel in NIVELES:
                for s in semillas_piloto:
                    c = recocido_simulado(instancias[n], TEMPERATURAS[nivel], N, s)
                    brechas.append(100 * (c - optimos[n]) / optimos[n])
            filas.append({"iteraciones": N, "tamano": n,
                          "brecha_media_pct": np.mean(brechas),
                          "pct_corridas_en_optimo": 100 * np.mean(np.array(brechas) < 1e-6)})
    return pd.DataFrame(filas)


# =============================================================================
# ANOVA (statsmodels; con respaldo manual equivalente en diseño balanceado)
# =============================================================================
def anova_dos_factores_manual(df):
    """Respaldo si no hay statsmodels. En un diseño balanceado coincide con anova_lm(typ=2)."""
    gm = df["costo"].mean()
    g_a, g_b = df.groupby("temperatura")["costo"], df.groupby("tamano")["costo"]
    g_ab = df.groupby(["temperatura", "tamano"])["costo"]
    ss_a = sum(len(g) * (g.mean() - gm) ** 2 for _, g in g_a)
    ss_b = sum(len(g) * (g.mean() - gm) ** 2 for _, g in g_b)
    ss_celdas = sum(len(g) * (g.mean() - gm) ** 2 for _, g in g_ab)
    ss_ab = ss_celdas - ss_a - ss_b
    ss_e = sum(((g - g.mean()) ** 2).sum() for _, g in g_ab)
    gl = {"C(temperatura)": g_a.ngroups - 1, "C(tamano)": g_b.ngroups - 1}
    gl["C(temperatura):C(tamano)"] = gl["C(temperatura)"] * gl["C(tamano)"]
    gl["Residual"] = len(df) - g_ab.ngroups
    ss = {"C(temperatura)": ss_a, "C(tamano)": ss_b, "C(temperatura):C(tamano)": ss_ab, "Residual": ss_e}
    tabla = pd.DataFrame({"sum_sq": ss, "df": gl})
    f = (tabla["sum_sq"] / tabla["df"]) / (ss_e / gl["Residual"])
    tabla["F"] = f
    tabla.loc["Residual", "F"] = np.nan
    tabla["PR(>F)"] = [stats.f.sf(f[k], gl[k], gl["Residual"]) if k != "Residual" else np.nan for k in tabla.index]
    return tabla


def anova_una_via_manual(sub):
    g = sub.groupby("temperatura")["costo"]
    gm = sub["costo"].mean()
    ss_a = sum(len(x) * (x.mean() - gm) ** 2 for _, x in g)
    ss_e = sum(((x - x.mean()) ** 2).sum() for _, x in g)
    gl_a, gl_e = g.ngroups - 1, len(sub) - g.ngroups
    f = (ss_a / gl_a) / (ss_e / gl_e)
    return pd.DataFrame({"sum_sq": [ss_a, ss_e], "df": [gl_a, gl_e], "F": [f, np.nan],
                         "PR(>F)": [stats.f.sf(f, gl_a, gl_e), np.nan]},
                        index=["C(temperatura)", "Residual"])


def anova_dos_factores(df, respuesta="costo"):
    """[REQUISITO: ANOVA de dos factores con interacción, statsmodels, typ=2, alfa = 0.05]"""
    d = df.copy()
    d["costo"] = d[respuesta]
    if HAY_STATSMODELS:
        modelo = ols('costo ~ C(temperatura) * C(tamano)', data=d).fit()
        tabla = sm.stats.anova_lm(modelo, typ=2)
    else:
        tabla = anova_dos_factores_manual(d)
    tabla["eta_cuadrado"] = tabla["sum_sq"] / tabla["sum_sq"].sum()
    return tabla


def anova_una_via(sub):
    """ANOVA de un factor (temperatura) dentro de un tamaño; None si no hay variación (efecto piso)."""
    if sub["costo"].nunique() == 1:
        return None
    if HAY_STATSMODELS:
        return sm.stats.anova_lm(ols('costo ~ C(temperatura)', data=sub).fit(), typ=2)
    return anova_una_via_manual(sub)


def kruskal_una_via(sub):
    """Prueba no paramétrica (Kruskal-Wallis) de la temperatura dentro de un tamaño, como apoyo
    porque los supuestos del ANOVA (normalidad, varianzas iguales) no se cumplen del todo."""
    if sub["costo"].nunique() == 1:          # todos los valores idénticos (efecto de piso)
        return np.nan
    return stats.kruskal(*[g["costo"].values for _, g in sub.groupby("temperatura")]).pvalue


# =============================================================================
# BOOTSTRAP de la mediana
# =============================================================================
def bootstrap_mediana(x, rng, B=N_BOOT):
    """[REQUISITO: mediana vía bootstrap con IC del 95% (percentiles 2.5 y 97.5)]"""
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(B, len(x)))      # remuestreo CON reemplazo, mismo tamaño
    medianas = np.median(x[idx], axis=1)
    return np.median(medianas), np.percentile(medianas, 2.5), np.percentile(medianas, 97.5)


def bootstrap_diferencia_pareada(x_otro, x_mejor, rng, B=N_BOOT):
    """IC95% de mediana(otro) - mediana(mejor), remuestreando las MISMAS semillas en ambos niveles
    (aprovecha el emparejamiento que dan los Common Random Numbers)."""
    a, b = np.asarray(x_otro, dtype=float), np.asarray(x_mejor, dtype=float)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    dif = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    return np.percentile(dif, 2.5), np.percentile(dif, 97.5)


# =============================================================================
# EXPERIMENTO PRINCIPAL
# =============================================================================
def correr_experimento(instancias):
    """[REQUISITO: diseño factorial completo 3 x 3 puntos de diseño x 10 réplicas = 90 corridas]"""
    filas = []
    for n in TAMANOS:
        for nivel in NIVELES:
            for s in SEMILLAS:
                costo = recocido_simulado(instancias[n], TEMPERATURAS[nivel], N_ITER, s)
                filas.append({"temperatura": nivel, "T0": TEMPERATURAS[nivel], "tamano": n,
                              "semilla": s, "costo": costo})
    return pd.DataFrame(filas)


# =============================================================================
# GRÁFICAS
# =============================================================================
def etiqueta_nivel(nivel):
    return f"{nivel}\n(T0 = {TEMPERATURAS[nivel]:.1f})"


def media_ic(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0


def grafica_efectos_principales(df):
    """[REQUISITO: gráfica de efectos principales (temperatura y tamaño)]"""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    # (a) efecto de la temperatura sobre el costo (promedio sobre los 3 tamaños)
    m = [media_ic(df[df.temperatura == nv]["costo"]) for nv in NIVELES]
    ax[0].errorbar(range(3), [x[0] for x in m], yerr=[x[1] for x in m], fmt="o-", capsize=5)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels([etiqueta_nivel(nv) for nv in NIVELES])
    ax[0].set_title("Efecto principal: temperatura"); ax[0].set_ylabel("costo medio (longitud del tour)")
    # (b) efecto del tamaño sobre el costo
    m = [media_ic(df[df.tamano == n]["costo"]) for n in TAMANOS]
    ax[1].errorbar(range(3), [x[0] for x in m], yerr=[x[1] for x in m], fmt="s-", color="tab:orange", capsize=5)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels([f"n = {n}" for n in TAMANOS])
    ax[1].set_title("Efecto principal: tamaño del problema"); ax[1].set_ylabel("costo medio (longitud del tour)")
    # (c) apoyo: efecto de la temperatura sobre la brecha % al óptimo (quita el efecto del tamaño)
    m = [media_ic(df[df.temperatura == nv]["brecha_pct"]) for nv in NIVELES]
    ax[2].errorbar(range(3), [x[0] for x in m], yerr=[x[1] for x in m], fmt="o-", color="tab:green", capsize=5)
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels([etiqueta_nivel(nv) for nv in NIVELES])
    ax[2].set_title("Apoyo: temperatura vs. brecha al óptimo"); ax[2].set_ylabel("brecha media (%)")
    for a in ax:
        a.grid(alpha=.3)
    fig.suptitle("Efectos principales (media ± 1.96·error estándar)")
    guardar_fig(fig, "t3_efectos_principales.png")


def grafica_interaccion(df):
    """[REQUISITO: gráfica de interacción temperatura x tamaño]"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, col, ylab, titulo in [
        (ax[0], "costo", "costo medio (longitud del tour)", "Interacción temperatura × tamaño (costo)"),
        (ax[1], "brecha_pct", "brecha media al óptimo (%)", "Misma interacción, como brecha al óptimo"),
    ]:
        for n, marc in zip(TAMANOS, ["o", "s", "^"]):
            m = [media_ic(df[(df.tamano == n) & (df.temperatura == nv)][col]) for nv in NIVELES]
            a.errorbar(range(3), [x[0] for x in m], yerr=[x[1] for x in m], marker=marc,
                       capsize=4, label=f"n = {n}")
        a.set_xticks(range(3)); a.set_xticklabels([etiqueta_nivel(nv) for nv in NIVELES])
        a.set_ylabel(ylab); a.set_title(titulo); a.grid(alpha=.3)
        a.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    guardar_fig(fig, "t3_interaccion.png")


def grafica_bootstrap(boot):
    """Mediana bootstrap con su IC95% por nivel de temperatura, una subgráfica por tamaño."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for a, n in zip(ax, TAMANOS):
        sub = boot[boot.tamano == n].set_index("temperatura").loc[NIVELES]
        mejor = sub["mediana_boot"].idxmin()
        valores = sub[["mediana_boot", "ic_inf", "ic_sup"]].values
        empate = np.allclose(valores, valores[0])           # los 3 niveles idénticos (efecto de piso)
        for i, nv in enumerate(NIVELES):
            fila = sub.loc[nv]
            color = "tab:green" if (nv == mejor and not empate) else "tab:blue"
            a.errorbar(i, fila.mediana_boot, yerr=[[fila.mediana_boot - fila.ic_inf], [fila.ic_sup - fila.mediana_boot]],
                       fmt="o", color=color, capsize=6, markersize=8)
        a.set_xticks(range(3)); a.set_xticklabels([etiqueta_nivel(nv) for nv in NIVELES])
        a.set_xlim(-0.5, 2.5)
        a.set_title(f"n = {n}  (empate exacto: efecto de piso)" if empate else f"n = {n}  (verde = mejor nivel)")
        a.set_ylabel("mediana bootstrap del costo"); a.grid(alpha=.3)
    fig.suptitle("Mediana bootstrap e IC 95% (10,000 remuestras)")
    guardar_fig(fig, "t3_bootstrap.png")


def grafica_cajas(df):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for a, n in zip(ax, TAMANOS):
        datos = [df[(df.tamano == n) & (df.temperatura == nv)]["costo"] for nv in NIVELES]
        a.boxplot(datos, tick_labels=[etiqueta_nivel(nv) for nv in NIVELES], widths=.5)
        for i, d in enumerate(datos, start=1):
            a.scatter(np.full(len(d), i) + np.linspace(-.12, .12, len(d)), d, s=12, alpha=.6, color="tab:blue")
        a.set_title(f"n = {n}"); a.set_ylabel("costo (longitud del tour)"); a.grid(alpha=.3, axis="y")
    fig.suptitle("Distribución de las 10 réplicas en cada punto de diseño")
    guardar_fig(fig, "t3_cajas.png")


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
def main():
    print("=" * 78)
    print("TAREA 3 - DOE sobre la temperatura del Recocido Simulado (Opción A)")
    print("=" * 78)
    print(f"Matrícula = {MATRICULA}   ->   r = {r:.6f}   (fijo para todo el experimento)")
    print(f"Rango de temperatura: [T_min, T_max] = [{T_MIN}, {T_MAX}]   (T_final = {T_FINAL})")
    print(f"{'nivel':8s} {'fracción':>10s} {'T0 real':>9s}")
    for nv in NIVELES:
        print(f"{nv:8s} {FRACCIONES[nv]:10.4f} {TEMPERATURAS[nv]:9.3f}")
    print(f"Semillas (las mismas en los 9 puntos de diseño): {SEMILLAS}")
    print(f"ANOVA con: {'statsmodels (ols + anova_lm)' if HAY_STATSMODELS else 'respaldo manual (statsmodels no instalado)'}")

    # [REQUISITO: una sola instancia fija por tamaño n]
    instancias = {n: crear_instancia(n) for n in TAMANOS}
    optimos = {n: round(optimo_held_karp(instancias[n]), 6) for n in TAMANOS}
    print("\nÓptimo exacto (Held-Karp) de cada instancia fija:",
          {n: round(optimos[n], 2) for n in TAMANOS})

    # ---------------- Piloto ----------------
    print("\n--- Estudio piloto (semillas distintas a las del experimento) ---")
    piloto = estudio_piloto(instancias, optimos)
    print(piloto.round(2).to_string(index=False))
    print(f"-> Presupuesto común elegido: N_ITER = {N_ITER} iteraciones por corrida.")
    print("   Razón: con 100 iteraciones n = 10 y n = 15 quedan lejos del piso (no todas las corridas llegan")
    print("   al óptimo) y del techo. n = 5 llega al óptimo en el 100% de las corridas con cualquier presupuesto")
    print("   probado, así que en n = 5 se espera un efecto de piso que se reportará como tal.")

    # ---------------- Experimento ----------------
    df = correr_experimento(instancias)
    df["brecha_pct"] = [100 * (c - optimos[n]) / optimos[n] for c, n in zip(df["costo"], df["tamano"])]
    assert len(df) == 90 and df.groupby(["temperatura", "tamano"]).size().eq(REPLICAS).all()
    print(f"\nCorridas totales: {len(df)} (3 temperaturas x 3 tamaños x {REPLICAS} réplicas)")
    print("\nCosto medio / mediana / desv. estándar por punto de diseño:")
    print(df.groupby(["tamano", "temperatura"])["costo"].agg(["mean", "median", "std"]).round(2).to_string())

    # ---------------- ANOVA ----------------
    tabla = anova_dos_factores(df)
    print("\n--- ANOVA de dos factores con interacción (variable: costo, alfa = 0.05) ---")
    print(tabla.round(4).to_string())
    p = {k: tabla.loc[k, "PR(>F)"] for k in ["C(temperatura)", "C(tamano)", "C(temperatura):C(tamano)"]}
    for k, v in p.items():
        print(f"  {k:28s} p = {v:.4g}  ->  {'SIGNIFICATIVO' if v < ALFA else 'no significativo'}")

    # Verificación de supuestos del ANOVA
    residuos = df["costo"] - df.groupby(["temperatura", "tamano"])["costo"].transform("mean")
    grupos = [g["costo"].values for _, g in df.groupby(["temperatura", "tamano"])]
    sw = stats.shapiro(residuos)
    lev = stats.levene(*grupos, center="median")
    print(f"\nSupuestos: Shapiro-Wilk de residuos p = {sw.pvalue:.4g} | Levene (9 celdas) p = {lev.pvalue:.4g}")
    tabla_gap = anova_dos_factores(df, respuesta="brecha_pct")
    print("\nANOVA complementario sobre la brecha % al óptimo (quita el efecto de escala del tamaño):")
    print(tabla_gap.round(4).to_string())

    # ---------------- Bootstrap de los 9 puntos ----------------
    nombres = {"C(temperatura)": "temperatura", "C(tamano)": "tamaño", "C(temperatura):C(tamano)": "interacción"}
    significativos = [nombres[k] for k, v in p.items() if v < ALFA]
    print(f"\nEfectos significativos (alfa = {ALFA}): {significativos if significativos else 'ninguno'}")

    # [REQUISITO: mediana bootstrap con IC 95% para los puntos de diseño cuyo efecto resultó
    #  significativo. Los 9 puntos pertenecen al efecto significativo (el tamaño) y además la tabla
    #  final del profesor pide reportarla por tamaño, así que se calcula en los 9 puntos.]
    rng = np.random.default_rng(SEMILLA_BOOT)
    filas, simples = [], {}
    for n in TAMANOS:
        sub = df[df.tamano == n]
        simples[n] = anova_una_via(sub)
        p_simple = np.nan if simples[n] is None else simples[n].loc["C(temperatura)", "PR(>F)"]
        for nv in NIVELES:
            x = sub[sub.temperatura == nv]["costo"].values
            med, lo, hi = bootstrap_mediana(x, rng)
            filas.append({"tamano": n, "temperatura": nv, "T0": TEMPERATURAS[nv],
                          "mediana_muestral": np.median(x), "media": x.mean(),
                          "mediana_boot": med, "ic_inf": lo, "ic_sup": hi})
    boot = pd.DataFrame(filas)
    print("\n--- Mediana bootstrap (10,000 remuestras) e IC 95% por punto de diseño ---")
    print(boot.round(3).to_string(index=False))

    # ---------------- Tabla resumen: mejor temperatura por tamaño ----------------
    resumen = []
    print("\n--- Resumen: mejor temperatura por tamaño ---")
    for n in TAMANOS:
        sub = boot[boot.tamano == n].sort_values(["mediana_boot", "media"])
        mejor = sub.iloc[0]
        x_mejor = df[(df.tamano == n) & (df.temperatura == mejor.temperatura)]["costo"].values
        comparaciones = {}
        for nv in NIVELES:
            if nv == mejor.temperatura:
                continue
            x_otro = df[(df.tamano == n) & (df.temperatura == nv)]["costo"].values
            comparaciones[nv] = bootstrap_diferencia_pareada(x_otro, x_mejor, rng)
        p_simple = np.nan if simples[n] is None else simples[n].loc["C(temperatura)", "PR(>F)"]
        p_kw = kruskal_una_via(df[df.tamano == n])
        distinguible = all(lo > 0 for lo, hi in comparaciones.values())
        empatados = [nv for nv in NIVELES if nv != mejor.temperatura and
                     np.allclose(boot[(boot.tamano == n) & (boot.temperatura == nv)][["mediana_boot", "ic_inf", "ic_sup"]].values,
                                 [[mejor.mediana_boot, mejor.ic_inf, mejor.ic_sup]])]
        resumen.append({"tamano": n, "mejor_nivel": mejor.temperatura, "T_optima": mejor.T0,
                        "mediana_boot": mejor.mediana_boot, "ic_inf": mejor.ic_inf, "ic_sup": mejor.ic_sup,
                        "p_anova_temperatura_en_n": p_simple, "p_kruskal_temperatura_en_n": p_kw,
                        "empate_exacto_con": ", ".join(empatados),
                        "mejor_distinguible_de_los_demas": distinguible,
                        **{f"dif_vs_{nv}_ic": f"[{lo:.2f}, {hi:.2f}]" for nv, (lo, hi) in comparaciones.items()}})
        print(f"n = {n:2d}: mejor = {mejor.temperatura:5s} (T0 = {mejor.T0:.2f}) | mediana boot = "
              f"{mejor.mediana_boot:.2f}  IC95% [{mejor.ic_inf:.2f}, {mejor.ic_sup:.2f}] | "
              f"p(ANOVA temp. en n) = {p_simple:.4g} | p(Kruskal) = {p_kw:.4g} | "
              f"distinguible de los demás por bootstrap pareado: {distinguible}")
        if empatados:
            print(f"        EMPATE EXACTO con: {empatados} (todas las corridas dan el mismo costo: efecto de piso)")
        for nv, (lo, hi) in comparaciones.items():
            print(f"        mediana({nv}) - mediana({mejor.temperatura}): IC95% [{lo:.2f}, {hi:.2f}]")

    # [REQUISITO: ¿cambia T* con n? ¿hay tendencia?] -> se discute en el reporte con estos datos
    print("\nT* por tamaño:", {x["tamano"]: round(float(x["T_optima"]), 2) for x in resumen})

    # ---------------- Gráficas ----------------
    print("\n--- Gráficas ---")
    grafica_efectos_principales(df)
    grafica_interaccion(df)
    grafica_bootstrap(boot)
    grafica_cajas(df)

    print(f"\nListo. Archivos en: {CARPETA}")


if __name__ == "__main__":
    main()
