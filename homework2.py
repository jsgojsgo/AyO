"""
Segundo parcial - Optimización de algoritmos (Ejercicios 1-5)

Uso:
    python ejercicios.py          # corre los 5 ejercicios
    python ejercicios.py 1 4      # corre solo los ejercicios 1 y 4

Requiere: matplotlib   (pip install matplotlib)
Las gráficas se guardan como PNG en la carpeta Downloads (Descargas) del usuario.

Los ejercicios 3, 4 y 5 comparten las mismas piezas del TSP (matriz de distancias,
longitud de tour, movimiento 2-opt, hill climbing), definidas una sola vez en la
sección "Bloques del TSP".
"""
import itertools
import math
import multiprocessing as mp
import random
import statistics as st
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # guarda a archivos (no necesita ventana)
import matplotlib.pyplot as plt

CARPETA_SALIDA = Path.home() / "Downloads"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)


def guardar(fig, nombre):
    ruta = CARPETA_SALIDA / nombre
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  [gráfica guardada] {ruta}")


# =============================================================================
# EJERCICIO 1 - Aceleración paralela (speedup) con procesos
# =============================================================================
def estimate_pi(seed: int, samples: int = 1_000_000) -> float:
    """Monte Carlo: fracción de puntos aleatorios dentro del cuarto de círculo unitario."""
    rng = random.Random(seed)
    dentro = sum(rng.random() ** 2 + rng.random() ** 2 <= 1.0 for _ in range(samples))
    return 4 * dentro / samples


def tiempo_con_procesos(n_procesos: int, semillas: list[int]) -> float:
    """Mismo trabajo total (todas las semillas) repartido entre n_procesos."""
    t0 = time.perf_counter()
    with mp.Pool(processes=n_procesos) as pool:
        pool.map(estimate_pi, semillas, chunksize=1)
    return time.perf_counter() - t0


def ejercicio_1(repeticiones: int = 3):
    print("\n=== Ejercicio 1: aceleración paralela ===")
    semillas = list(range(8))            # 8 corridas independientes = trabajo fijo
    procesos = [1, 2, 4, 8]
    nucleos = mp.cpu_count()
    tiempos = []
    for k in procesos:
        t = st.mean(tiempo_con_procesos(k, semillas) for _ in range(repeticiones))
        tiempos.append(t)
    speedup = [tiempos[0] / t for t in tiempos]

    print(f"  Núcleos disponibles: {nucleos}")
    print(f"  {'procesos':>8s} {'tiempo (s)':>11s} {'speedup':>8s} {'eficiencia':>11s}")
    for k, t, s in zip(procesos, tiempos, speedup):
        print(f"  {k:8d} {t:11.3f} {s:8.2f} {s / k:11.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(procesos, tiempos, "o-", label="medido")
    ax[0].plot(procesos, [tiempos[0] / k for k in procesos], "k--", alpha=.5, label="ideal (t₁ / k)")
    ax[0].set_xlabel("número de procesos"); ax[0].set_ylabel("tiempo (s)")
    ax[0].set_title("Tiempo de ejecución (8 × estimate_pi)")
    ax[0].set_xticks(procesos); ax[0].grid(alpha=.3); ax[0].legend()

    ax[1].plot(procesos, speedup, "o-", label="speedup medido")
    ax[1].plot(procesos, procesos, "k--", alpha=.6, label="speedup ideal (lineal)")
    ax[1].plot(procesos, [min(k, nucleos) for k in procesos], "r:", alpha=.7,
               label=f"límite por núcleos ({nucleos})")
    ax[1].set_xlabel("número de procesos"); ax[1].set_ylabel("speedup  (t₁ / t)")
    ax[1].set_title("Aceleración vs. número de procesos")
    ax[1].set_xticks(procesos); ax[1].grid(alpha=.3); ax[1].legend()
    guardar(fig, "ej1_speedup.png")


# =============================================================================
# EJERCICIO 2 - Mochila voraz: menor peso primero vs. valor/peso
# =============================================================================
def mochila_voraz(valores, pesos, W, clave):
    """Ordena los objetos por `clave` (ascendente) y toma cada uno que aún quepa."""
    orden = sorted(range(len(valores)), key=clave)
    elegidos, peso_usado, valor_total = [], 0, 0
    for i in orden:
        if peso_usado + pesos[i] <= W:
            elegidos.append(i)
            peso_usado += pesos[i]
            valor_total += valores[i]
    return sorted(elegidos), peso_usado, valor_total


def voraz_menor_peso(valores, pesos, W):
    return mochila_voraz(valores, pesos, W, clave=lambda i: pesos[i])


def voraz_valor_peso(valores, pesos, W):
    return mochila_voraz(valores, pesos, W, clave=lambda i: -valores[i] / pesos[i])


def mochila_dp(valores, pesos, W):
    """Programación dinámica (óptimo), solo como referencia."""
    n = len(valores)
    K = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        v, w = valores[i - 1], pesos[i - 1]
        for c in range(W + 1):
            K[i][c] = K[i - 1][c]
            if w <= c:
                K[i][c] = max(K[i][c], v + K[i - 1][c - w])
    elegidos, c = [], W
    for i in range(n, 0, -1):
        if K[i][c] != K[i - 1][c]:
            elegidos.append(i - 1)
            c -= pesos[i - 1]
    elegidos = sorted(elegidos)
    return elegidos, sum(pesos[i] for i in elegidos), K[n][W]


def ejercicio_2():
    print("\n=== Ejercicio 2: mochila voraz (menor peso vs. valor/peso) ===")
    nombres = ["A", "B", "C", "D"]
    valores = [2, 3, 100, 8]
    pesos = [1, 2, 9, 4]
    W = 10

    print(f"  Instancia (capacidad W = {W}):")
    print("   objeto  valor  peso  valor/peso")
    for n, v, w in zip(nombres, valores, pesos):
        print(f"   {n:>6s}  {v:5d}  {w:4d}  {v / w:10.2f}")

    metodos = {
        "Voraz: menor peso":  voraz_menor_peso(valores, pesos, W),
        "Voraz: valor/peso":  voraz_valor_peso(valores, pesos, W),
        "Óptimo (PD)":        mochila_dp(valores, pesos, W),
    }
    filas = []
    print(f"\n  {'método':20s} {'objetos':>14s} {'peso usado':>11s} {'valor total':>12s}")
    for m, (ch, peso, valor) in metodos.items():
        objs = ", ".join(nombres[i] for i in ch)
        filas.append([m, objs, f"{peso} / {W}", str(valor)])
        print(f"  {m:20s} {objs:>14s} {peso:>8d}/{W:<2d} {valor:12d}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    ax[0].axis("off")
    tabla = ax[0].table(cellText=filas,
                        colLabels=["Método", "Objetos elegidos", "Capacidad usada", "Valor total"],
                        loc="center", cellLoc="center")
    tabla.auto_set_font_size(False); tabla.set_fontsize(10); tabla.scale(1, 1.8)
    ax[0].set_title(f"Instancia: v = {valores}, w = {pesos}, W = {W}", fontsize=10)

    etiquetas = ["Menor peso", "Valor/peso", "Óptimo (PD)"]
    x = range(len(etiquetas))
    pesos_usados = [v[1] for v in metodos.values()]
    valores_tot = [v[2] for v in metodos.values()]
    ax[1].bar([i - 0.2 for i in x], valores_tot, 0.4, label="Valor total")
    ax[1].bar([i + 0.2 for i in x], pesos_usados, 0.4, label="Capacidad usada")
    ax[1].axhline(W, color="r", ls="--", alpha=.6, label=f"capacidad W = {W}")
    for i, (v, p) in enumerate(zip(valores_tot, pesos_usados)):
        ax[1].text(i - 0.2, v + 1, str(v), ha="center", fontsize=9)
        ax[1].text(i + 0.2, p + 1, str(p), ha="center", fontsize=9)
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels(etiquetas)
    ax[1].set_title("Valor total y capacidad usada"); ax[1].legend(fontsize=8)
    guardar(fig, "ej2_mochila.png")


# =============================================================================
# Bloques del TSP (compartidos por los ejercicios 3, 4 y 5)
# =============================================================================
def crear_ciudades(n, semilla, lado=100.0):
    rng = random.Random(semilla)
    return [(rng.uniform(0, lado), rng.uniform(0, lado)) for _ in range(n)]


def matriz_distancias(pts):
    return [[math.dist(p, q) for q in pts] for p in pts]


def longitud_tour(D, tour):
    return sum(D[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def vecino_mas_cercano(D, inicio):
    """Heurística constructiva: ir siempre a la ciudad no visitada más cercana."""
    tour, faltan = [inicio], set(range(len(D))) - {inicio}
    while faltan:
        sig = min(faltan, key=lambda j: D[tour[-1]][j])
        tour.append(sig)
        faltan.remove(sig)
    return tour


def delta_2opt(D, t, i, k):
    """Cambio en la longitud si se invierte t[i..k] (se reemplazan las aristas (i-1,i) y (k,k+1))."""
    a, b, c, d = t[i - 1], t[i], t[k], t[(k + 1) % len(t)]
    return D[a][c] + D[b][d] - D[a][b] - D[c][d]


def hill_climbing_2opt(D, tour):
    """
    Hill climbing (ascenso más pronunciado) con vecindario 2-opt: revisa todos los pares (i, k),
    aplica la mejor inversión que acorte el tour y repite hasta que ninguna mejore.
    Regresa (tour, número_de_mejoras, historial_de_longitudes).
    """
    t = tour[:]
    n = len(t)
    actual = longitud_tour(D, t)
    historial = [actual]
    mejoras = 0
    while True:
        mejor_delta, mejor_ik = -1e-9, None
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                delta = delta_2opt(D, t, i, k)
                if delta < mejor_delta:
                    mejor_delta, mejor_ik = delta, (i, k)
        if mejor_ik is None:             # óptimo local alcanzado
            return t, mejoras, historial
        i, k = mejor_ik
        t[i:k + 1] = t[i:k + 1][::-1]
        actual += mejor_delta
        mejoras += 1
        historial.append(actual)


def tour_aleatorio(n, rng):
    return rng.sample(range(n), n)


# =============================================================================
# EJERCICIO 3 - Búsqueda local: 2-opt desde inicio aleatorio vs. vecino más cercano
# =============================================================================
def ejercicio_3(n_ciudades: int = 50, n_instancias: int = 30):
    print("\n=== Ejercicio 3: búsqueda local (2-opt) ===")
    escenarios = ["Inicio aleatorio", "Vecino más cercano\n(nodo aleatorio)"]
    mejoras = {e: [] for e in escenarios}
    finales = {e: [] for e in escenarios}
    iniciales = {e: [] for e in escenarios}

    for inst in range(n_instancias):
        D = matriz_distancias(crear_ciudades(n_ciudades, semilla=inst))
        rng = random.Random(1000 + inst)
        inicios = {
            escenarios[0]: tour_aleatorio(n_ciudades, rng),
            escenarios[1]: vecino_mas_cercano(D, rng.randrange(n_ciudades)),
        }
        for e, inicio in inicios.items():
            _, n_mej, hist = hill_climbing_2opt(D, inicio)
            mejoras[e].append(n_mej)
            iniciales[e].append(hist[0])
            finales[e].append(hist[-1])

    print(f"  {n_instancias} instancias aleatorias de {n_ciudades} ciudades")
    print(f"  {'escenario':28s} {'L inicial':>10s} {'L final':>9s} {'mejoras (media)':>16s} {'mejoras (mediana)':>18s}")
    for e in escenarios:
        print(f"  {e.replace(chr(10), ' '):28s} {st.mean(iniciales[e]):10.1f} {st.mean(finales[e]):9.1f}"
              f" {st.mean(mejoras[e]):16.1f} {st.median(mejoras[e]):18.1f}")
    nn_gana = sum(a < b for a, b in zip(finales[escenarios[1]], finales[escenarios[0]]))
    print(f"  El óptimo local con vecino más cercano fue más corto en {nn_gana}/{n_instancias} instancias")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    colores = ["tab:green", "tab:blue"]
    for a, datos, titulo, etiqueta in [
        (ax[0], mejoras, "Mejoras necesarias para llegar al óptimo local", "número de mejoras 2-opt"),
        (ax[1], finales, "Calidad del óptimo local alcanzado", "longitud final del tour"),
    ]:
        bp = a.boxplot([datos[e] for e in escenarios], tick_labels=escenarios, patch_artist=True)
        for caja, c in zip(bp["boxes"], colores):
            caja.set_facecolor(c); caja.set_alpha(.5)
        a.set_title(titulo); a.set_ylabel(etiqueta); a.grid(alpha=.3, axis="y")
    guardar(fig, "ej3_2opt_inicios.png")


# =============================================================================
# EJERCICIO 4 - Recocido simulado: esquemas de enfriamiento
# =============================================================================
def recocido_simulado(D, inicio, esquema, n_pasos, rng, registrar_cada=None):
    """
    SA para TSP. Movimiento = 2-opt aleatorio. Objetivo = longitud del tour (minimizar).
    `esquema(paso, n_pasos)` regresa la temperatura. Regresa (tour, historial, mejor_longitud).
    historial = lista de (iteración, costo_actual, mejor_costo, temperatura).
    """
    t = inicio[:]
    n = len(t)
    actual = longitud_tour(D, t)
    mejor = actual
    historial = []
    for paso in range(n_pasos):
        T = esquema(paso, n_pasos)
        i, k = sorted(rng.sample(range(1, n), 2))
        delta = delta_2opt(D, t, i, k)
        if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
            t[i:k + 1] = t[i:k + 1][::-1]
            actual += delta
            if actual < mejor:
                mejor = actual
        if registrar_cada and paso % registrar_cada == 0:
            historial.append((paso, actual, mejor, T))
    return t, historial, mejor


def crear_esquemas(T0, Tf):
    """Esquemas de enfriamiento (lineal y exponencial van de T0 a Tf; el logarítmico enfría más lento)."""
    def lineal(s, N):
        return T0 + (Tf - T0) * s / N

    def exponencial(s, N):
        return T0 * (Tf / T0) ** (s / N)

    def logaritmico(s, N):
        # esquema logarítmico clásico T = c / ln(t + 2): enfría muy lento y NO llega a Tf
        return T0 * math.log(2) / math.log(s + 2)

    return {
        "Lineal": lineal,
        "Exponencial": exponencial,
        "Logarítmico": logaritmico,
        "Hill climbing (T = 0)": lambda s, N: 0.0,
    }


def ejercicio_4(n_ciudades: int = 40, n_pasos: int = 30_000, n_corridas: int = 10):
    print("\n=== Ejercicio 4: recocido simulado ===")
    D = matriz_distancias(crear_ciudades(n_ciudades, semilla=3))
    T0, Tf = 10.0, 0.01
    esquemas = crear_esquemas(T0, Tf)
    registrar = 50

    # ---- 1) Curvas de convergencia (promedio de n_corridas corridas, inicio aleatorio) ----
    print(f"  Convergencia: {n_corridas} corridas por esquema, {n_pasos:,} iteraciones cada una")
    promedios = {}
    for nombre, esq in esquemas.items():
        hists = []
        for s in range(n_corridas):
            inicio = tour_aleatorio(n_ciudades, random.Random(s))
            _, h, _ = recocido_simulado(D, inicio, esq, n_pasos, random.Random(100 + s), registrar)
            hists.append(h)
        pasos = [r[0] for r in hists[0]]
        promedios[nombre] = {
            "pasos": pasos,
            "actual": [st.mean(h[j][1] for h in hists) for j in range(len(pasos))],
            "mejor": [st.mean(h[j][2] for h in hists) for j in range(len(pasos))],
            "T": [hists[0][j][3] for j in range(len(pasos))],
        }
        print(f"   {nombre:22s} mejor costo final (media) = {promedios[nombre]['mejor'][-1]:7.1f}")

    finales = [p["mejor"][-1] for p in promedios.values()]
    lo, hi = min(finales) * 0.97, min(finales) * 1.5       # zoom: ignora el tour inicial enorme
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for nombre, p in promedios.items():
        estilo = "k--" if nombre.startswith("Hill") else "-"
        kw = {} if nombre.startswith("Hill") else {}
        ax[0].plot(p["pasos"], p["mejor"], estilo, label=nombre, **kw)
        if not nombre.startswith("Hill"):
            ax[1].plot(p["T"], p["actual"], label=nombre)
            ax[2].plot(p["pasos"], p["T"], label=nombre)
    ax[0].set_ylim(lo, hi)
    ax[0].set_xlabel("iteraciones"); ax[0].set_ylabel("costo (longitud del tour)")
    ax[0].set_title("Convergencia: mejor costo vs. iteraciones"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].set_xscale("log"); ax[1].invert_xaxis(); ax[1].set_ylim(lo, hi * 1.4)
    ax[1].set_xlabel("temperatura T (se enfría hacia la derecha)"); ax[1].set_ylabel("costo de la solución actual")
    ax[1].set_title("Costo actual vs. temperatura"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[2].set_yscale("log"); ax[2].set_xlabel("iteraciones"); ax[2].set_ylabel("temperatura T")
    ax[2].set_title("Esquemas de enfriamiento"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    guardar(fig, "ej4_convergencia.png")

    # ---- 2) Tiempo de ejecución vs. calidad final (variando el presupuesto de iteraciones) ----
    presupuestos = [5_000, 10_000, 20_000, 40_000]
    n_rep = 5
    print(f"\n  Tiempo vs. calidad: presupuestos {presupuestos}, {n_rep} corridas por punto")
    print(f"  {'esquema':22s} {'iteraciones':>12s} {'tiempo (s)':>11s} {'mejor costo':>12s}")
    resultados = {nombre: [] for nombre in esquemas}
    for nombre, esq in esquemas.items():
        for N in presupuestos:
            tiempos, costos = [], []
            for s in range(n_rep):
                inicio = tour_aleatorio(n_ciudades, random.Random(s))
                t0 = time.perf_counter()
                _, _, mejor = recocido_simulado(D, inicio, esq, N, random.Random(200 + s))
                tiempos.append(time.perf_counter() - t0)
                costos.append(mejor)
            resultados[nombre].append((N, st.mean(tiempos), st.mean(costos)))
            print(f"  {nombre:22s} {N:12,d} {st.mean(tiempos):11.3f} {st.mean(costos):12.1f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for nombre, pts in resultados.items():
        estilo = "k--s" if nombre.startswith("Hill") else "o-"
        ax.plot([p[1] for p in pts], [p[2] for p in pts], estilo, label=nombre)
        for N, tmp, cst in pts:
            ax.annotate(f"{N // 1000}k", (tmp, cst), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("tiempo de ejecución (s)"); ax.set_ylabel("calidad final: mejor costo (menor = mejor)")
    ax.set_title("Tiempo de ejecución vs. calidad de la solución\n(cada punto = presupuesto de iteraciones)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    guardar(fig, "ej4_tiempo_vs_calidad.png")


# =============================================================================
# EJERCICIO 5 - Razón de aproximación: MST + comparación con métodos previos
# =============================================================================
def mst_prim(D):
    """Prim desde la ciudad 0. Regresa (hijos, peso_total d)."""
    n = len(D)
    mejor = {v: (D[0][v], 0) for v in range(1, n)}
    hijos = {v: [] for v in range(n)}
    total = 0.0
    while mejor:
        v = min(mejor, key=lambda u: mejor[u][0])
        w, padre = mejor.pop(v)
        total += w
        hijos[padre].append(v)
        for u in mejor:
            if D[v][u] < mejor[u][0]:
                mejor[u] = (D[v][u], v)
    return hijos, total


def preorden(hijos, raiz=0):
    return [raiz] + [c for h in hijos[raiz] for c in preorden(hijos, h)]


def held_karp(D):
    """Longitud del tour óptimo exacto por programación dinámica, O(n² 2ⁿ)."""
    n = len(D)
    C = {(1 << k, k): D[0][k] for k in range(1, n)}
    for tam in range(2, n):
        for sub in itertools.combinations(range(1, n), tam):
            bits = sum(1 << k for k in sub)
            for k in sub:
                previo = bits & ~(1 << k)
                C[(bits, k)] = min(C[(previo, m)] + D[m][k] for m in sub if m != k)
    completo = (1 << n) - 2
    return min(C[(completo, k)] + D[k][0] for k in range(1, n))


def ejercicio_5(n_instancias: int = 100, n_ciudades: int = 9, pasos_sa: int = 4_000):
    print("\n=== Ejercicio 5: razón de aproximación (MST) ===")
    lado = 10.0
    esq_sa = crear_esquemas(T0=lado / 10, Tf=lado / 10_000)["Exponencial"]   # misma escala que el Ej. 4
    nombres = ["MST puro", "MST + 2-opt", "2-opt (inicio aleatorio)",
               "2-opt (vecino más cercano)", "Recocido simulado"]
    costos = {n: [] for n in nombres}
    optimos, ds = [], []

    for inst in range(n_instancias):
        D = matriz_distancias(crear_ciudades(n_ciudades, semilla=500 + inst, lado=lado))
        rng = random.Random(7000 + inst)
        hijos, d = mst_prim(D)                       # d = peso del MST
        tour_mst = preorden(hijos)
        L_mst = longitud_tour(D, tour_mst)
        opt = held_karp(D)
        assert d <= opt + 1e-9 and opt <= L_mst + 1e-9 and L_mst <= 2 * d + 1e-9, "falla d <= OPT <= MST <= 2d"

        costos["MST puro"].append(L_mst)
        costos["MST + 2-opt"].append(hill_climbing_2opt(D, tour_mst)[2][-1])
        costos["2-opt (inicio aleatorio)"].append(hill_climbing_2opt(D, tour_aleatorio(n_ciudades, rng))[2][-1])
        costos["2-opt (vecino más cercano)"].append(
            hill_climbing_2opt(D, vecino_mas_cercano(D, rng.randrange(n_ciudades)))[2][-1])
        costos["Recocido simulado"].append(
            recocido_simulado(D, tour_aleatorio(n_ciudades, rng), esq_sa, pasos_sa, rng)[2])
        optimos.append(opt)
        ds.append(d)
    print("  Verificado en todas las instancias:  d  <=  ÓPTIMO  <=  tour MST  <=  2d")

    razones = {n: [c / o for c, o in zip(costos[n], optimos)] for n in nombres}
    print(f"\n  {'método':28s} {'razón media':>12s} {'mediana':>8s} {'máxima':>8s} {'% óptimo':>9s}")
    for n in nombres:
        r = razones[n]
        pct = 100 * sum(x <= 1 + 1e-9 for x in r) / len(r)
        print(f"  {n:28s} {st.mean(r):12.3f} {st.median(r):8.3f} {max(r):8.3f} {pct:8.0f}%")
    pos = lambda xs: st.mean((x - d) / d for x, d in zip(xs, ds))
    print(f"\n  Posición media en el rango [d, 2d] (0 = d, 1 = 2d): MST = {pos(costos['MST puro']):.2f}, "
          f"MST + 2-opt = {pos(costos['MST + 2-opt']):.2f}, óptimo = {pos(optimos):.2f}")

    # ---- Visualización 1: razón de aproximación del MST ----
    r_mst, r_2o = razones["MST puro"], razones["MST + 2-opt"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = range(1, n_instancias + 1)
    ax[0].scatter(xs, r_mst, s=14, label="MST puro")
    ax[0].scatter(xs, r_2o, s=14, label="MST + 2-opt")
    ax[0].axhline(2, color="r", ls="--", label="límite teórico (2)")
    ax[0].axhline(1, color="k", ls=":", label="óptimo (1)")
    ax[0].set_ylim(0.9, 2.1)
    ax[0].set_xlabel("instancia"); ax[0].set_ylabel("C obtenido / C óptimo")
    ax[0].set_title(f"Razón de aproximación ({n_ciudades} ciudades)"); ax[0].legend(fontsize=8)
    ax[1].hist(r_mst, bins=20, alpha=.7, label="MST puro")
    ax[1].hist(r_2o, bins=20, alpha=.7, label="MST + 2-opt")
    ax[1].axvline(2, color="r", ls="--", label="límite teórico (2)")
    ax[1].axvline(1, color="k", ls=":")
    ax[1].set_xlim(0.95, 2.05)
    ax[1].set_xlabel("C obtenido / C óptimo"); ax[1].set_ylabel("número de instancias")
    ax[1].set_title("Histograma de la razón de aproximación"); ax[1].legend(fontsize=8)
    guardar(fig, "ej5_razon_aproximacion.png")

    # ---- Visualización 2: comparación de costo entre métodos ----
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bp = ax.boxplot([razones[n] for n in nombres],
                    tick_labels=[n.replace(" (", "\n(") for n in nombres], patch_artist=True)
    for caja, c in zip(bp["boxes"], ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]):
        caja.set_facecolor(c); caja.set_alpha(.5)
    ax.axhline(1, color="k", ls=":", label="óptimo (1)")
    ax.set_ylabel("costo obtenido / costo óptimo")
    ax.set_title(f"Comparación de métodos sobre {n_instancias} instancias de {n_ciudades} ciudades")
    ax.grid(alpha=.3, axis="y"); ax.legend()
    guardar(fig, "ej5_comparacion_metodos.png")


# =============================================================================
if __name__ == "__main__":      # este "if" es necesario para multiprocessing en Windows/macOS
    todos = {"1": ejercicio_1, "2": ejercicio_2, "3": ejercicio_3, "4": ejercicio_4, "5": ejercicio_5}
    elegidos = sys.argv[1:] or list(todos)
    for e in elegidos:
        todos[e]()
    print(f"\nListo. Las gráficas están en: {CARPETA_SALIDA}")
