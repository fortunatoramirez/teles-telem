# Práctica 4 — Segmentación de fonocardiogramas (PCG) con **Entropía de Shannon normalizada**

**Serie:** Análisis → **Segmentación** → (Características → Dataset → ML → Evaluación)

> Objetivo de esta práctica (Parte 1): detectar y segmentar **ciclos cardiacos** en señales de audio (PCG) usando la **envolvente de Entropía de Shannon normalizada** + filtrado + detección de extremos. Al final tendrás los **límites de cada ciclo** listos para la siguiente práctica (extracción de características cepstrales).

---

## Datos y recursos

* Biblioteca de sonidos cardiacos (soplos, clicks, etc.):
  **Heart Sound & Murmur Library — University of Michigan**
  [https://open.umich.edu/find/open-educational-resources/medical/heart-sound-murmur-library](https://open.umich.edu/find/open-educational-resources/medical/heart-sound-murmur-library)

> Descarga al menos 3 audios **diferentes** (normal y patológicos) para comparar.

---

# BLOQUE 0 · Carga de datos y pre-procesamiento mínimo

### Teoría breve

Una señal de PCG $x[n]$ suele estar en rango $[-1, 1]$ o $[-\!32768,32767]$ si es entero. Para análisis robusto conviene trabajar en **mono** y en un rango **normalizado**. Usaremos una ventana corta (p. ej., 2 s) para visualizar con claridad.

### Código (MATLAB/Octave)

```matlab
clear; close all; clc;

% === 0.1 Leer audio (ajusta la ruta a tus archivos) ===
% Ejemplos del repositorio de Michigan (cambia por tus rutas):
% [x, fs] = audioread('data/09_Apex_HoloSysMur_Supine_Bell.mp3');
% [x, fs] = audioread('data/02_Apex_SplitS1_Supine_Bell.mp3');
[x, fs] = audioread('data/07_Apex_MidSysMur_Supine_Bell.mp3');

% === 0.2 Si es estéreo, convertir a mono ===
if size(x,2) > 1
    x = mean(x, 2);
end

% === 0.3 Normalizar a [-1, 1] ===
a = -1; b = 1;
x = (x - min(x)) / max(eps, (max(x) - min(x)));  % [0,1]
x = x * (b - a) + a;                              % [-1,1]

% === 0.4 Ventana de trabajo (2 s para empezar) ===
t = (0:length(x)-1)/fs;
max_time = 2;                         % segundos
sel = t <= max_time;
x = x(sel);  t = t(sel);
```

---

# BLOQUE 1 · Entropía de Shannon **normalizada** (envolvente básica)

### Teoría (con ecuaciones)

Sea la señal $x[n]$ y su magnitud $s[n]=|x[n]|$. Definimos una **probabilidad instantánea normalizada**:

$$
p[n] \;=\; \frac{s[n]}{\max_k s[k] + \varepsilon} \;\;\in [0,1]
$$

donde $\varepsilon$ evita división por cero.

La **energía/entropía de Shannon** puntual (forma habitual en PCG) puede escribirse como:

$$
E[n] \;=\; -\,p[n]\;\log_b\!\big(p[n] + \varepsilon\big)
$$

con base $b=\mathrm{e}$ o $b=10$. (En literatura también aparece $E[n]= -\,p^2[n]\log\!\big(p^2[n]\big)$; ambas son **monótonamente relacionadas** y sirven para realzar S1/S2).
Después **estandarizamos** y normalizamos a $[0,1]$ para obtener una **envolvente** más estable.

### Código

```matlab
% === 1.1 Probabilidad normalizada p[n] ===
p = abs(x);
p = p ./ max(eps, max(p));

% === 1.2 Entropía/energía de Shannon puntual (log10) ===
E = -p .* log10(p + eps);     % variante equivalente a -|x|log|x|

% === 1.3 Estandarización y normalización a [0,1] ===
E_z  = (E - mean(E)) / (std(E) + eps);
Env0 = (E_z - min(E_z)) / max(eps, (max(E_z) - min(E_z)));   % envolvente base [0,1]
```

---

# BLOQUE 2 · Suavizado de la envolvente (pasa-bajas)

### Teoría

El corazón late \~1–2 Hz; S1 y S2 están separados por el **silencio sistólico/diastólico**. El **suavizado pasa-bajas** sobre la envolvente (no sobre $x[n]$) elimina fluctuaciones espurias y resalta la morfología de ciclo. Un **Butterworth** de 4º orden con $f_c\approx 8\text{–}12\,\mathrm{Hz}$ funciona bien.

$$
H(z) = \frac{B(z)}{A(z)}, \quad \text{con } \omega_c = 2\pi \frac{f_c}{f_s}
$$

### Código

```matlab
% === 2.1 LPF sobre la envolvente (no sobre x) ===
fc = 10;                                         % corte ~10 Hz
[b,a] = butter(4, fc/(fs/2), 'low');
Env = filtfilt(b, a, Env0);

% === 2.2 Normalización final a [0,1] ===
Env = (Env - min(Env)) / max(eps, (max(Env) - min(Env)));

% Visual rápido
figure('Name','Señal & Envolvente (Shannon LPF)');
subplot(2,1,1); plot(t, x, 'k'); grid on;
title('Señal PCG (normalizada)'); xlabel('Tiempo (s)'); ylabel('Amplitud');

subplot(2,1,2); plot(t, Env, 'b'); grid on;
title('Envolvente de Shannon (LPF)'); xlabel('Tiempo (s)'); ylabel('Amplitud norm.');
```

---

# BLOQUE 3 · Extremos locales vía derivada (mínimos y máximos)

### Teoría

Con la envolvente $\text{Env}[n]$, calculamos la **derivada discreta** $d[n]=\text{Env}[n]-\text{Env}[n-1]$.
Un **cambio de signo** en $d[n]$ localiza **mínimos** ($d[n]\!<\!0$ y $d[n+1]\!>\!0$) y **máximos** ($d[n]\!>\!0$ y $d[n+1]\!<\!0$).

### Código

```matlab
% === 3.1 Derivada discreta y detección de cambios de signo ===
d = diff(Env);
idx_ext = []; tipo = [];  % tipo: 1 = min, 2 = max

for i = 1:length(d)-1
    if d(i) < 0 && d(i+1) > 0
        idx_ext(end+1) = i+1; tipo(end+1) = 1;  % mínimo
    elseif d(i) > 0 && d(i+1) < 0
        idx_ext(end+1) = i+1; tipo(end+1) = 2;  % máximo
    end
end

% Visual
figure('Name','Envolvente con min/máx');
plot(t, Env, 'b'); hold on; grid on;
plot(t(idx_ext(tipo==1)), Env(idx_ext(tipo==1)), 'go', 'DisplayName','Mínimos');
plot(t(idx_ext(tipo==2)), Env(idx_ext(tipo==2)), 'ro', 'DisplayName','Máximos');
legend; xlabel('Tiempo (s)'); ylabel('Amplitud norm.');
title('Extremos locales sobre la envolvente');
```

---

# BLOQUE 4 · Tripletes mín–máx–mín y **área** del triángulo

### Teoría

Un **evento** S1 o S2 suele coincidir con un **máximo** flanqueado por **mínimos** en la envolvente.
Modelamos cada candidato como un **triángulo** en el plano (tiempo, amplitud) con vértices $(t_1,y_1)$, $(t_2,y_2)$, $(t_3,y_3)$.
El **área** es:

$$
A \;=\; \frac{1}{2}\,\big|\,x_1 (y_2 - y_3) + x_2 (y_3 - y_1) + x_3 (y_1 - y_2)\,\big|
$$

Triángulos con **mayor área** $\Rightarrow$ eventos más **prominentes**.

### Código

```matlab
% === 4.1 Construcción de tripletes mín–máx–mín ===
tri_samp = []; tri_time = []; tri_amp = [];
for k = 1:length(tipo)-2
    if tipo(k)==1 && tipo(k+1)==2 && tipo(k+2)==1
        i1 = idx_ext(k); i2 = idx_ext(k+1); i3 = idx_ext(k+2);
        tri_samp(end+1,:) = [i1,i2,i3];                  %#ok<SAGROW>
        tri_time(end+1,:) = [t(i1), t(i2), t(i3)];       %#ok<SAGROW>
        tri_amp(end+1,:)  = [Env(i1), Env(i2), Env(i3)]; %#ok<SAGROW>
    end
end

% === 4.2 Área de cada triángulo ===
areas = zeros(size(tri_time,1),1);
for i = 1:size(tri_time,1)
    x1=tri_time(i,1); y1=tri_amp(i,1);
    x2=tri_time(i,2); y2=tri_amp(i,2);
    x3=tri_time(i,3); y3=tri_amp(i,3);
    areas(i) = 0.5 * abs( x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2) );
end

% Visual
figure('Name','Triángulos mín–máx–mín');
plot(t, Env, 'g', 'LineWidth', 1.2); hold on; grid on;
for i = 1:size(tri_time,1)
    tx = [tri_time(i,:), tri_time(i,1)];
    ty = [tri_amp(i,:),  tri_amp(i,1)];
    plot(tx, ty, 'r-', 'LineWidth', 1.0);
end
plot(t(idx_ext), Env(idx_ext), 'ko', 'MarkerSize', 4);
xlabel('Tiempo (s)'); ylabel('Amplitud norm.');
title('Envolvente y triángulos candidatos');
```

---

# BLOQUE 5 · Selección de eventos prominentes y **ciclos** cardiacos

### Teoría

1. **Filtra** triángulos por área $A > \bar{A}$ (umbral simple).
2. Convierte cada triángulo en un **ciclo** provisional usando $[t_{\min1},\, t_{\min2}]$.
3. Restringe por **fisiología**:

   * Intervalo RR $\in [0.3,\,1.5]\,\mathrm{s}$ (≈ 40–200 lpm).
   * Sin solapes.

### Código

```matlab
% === 5.1 Selección por área (umbral: promedio) ===
Amed = mean(areas);
mask_big = areas > Amed;

% === 5.2 Propuesta de ciclos: [min1, min2] de cada triángulo grande ===
ciclos_idx = tri_samp(mask_big, [1 3]);  % índices
ciclos_idx = sortrows(ciclos_idx, 1);

% === 5.3 Limpieza por duración fisiológica y no solape ===
minRR = round(0.30*fs);   % 0.30 s
maxRR = round(1.50*fs);   % 1.50 s
ciclos_ref = [];
for i = 1:size(ciclos_idx,1)
    L = ciclos_idx(i,2) - ciclos_idx(i,1);
    if L >= minRR && L <= maxRR
        if isempty(ciclos_ref) || ciclos_idx(i,1) > ciclos_ref(end,2)
            ciclos_ref = [ciclos_ref; ciclos_idx(i,:)]; %#ok<AGROW>
        end
    end
end

% === 5.4 Visual: sombrear ciclos en la envolvente ===
figure('Name','Ciclos cardiacos detectados');
plot(t, Env, 'k'); grid on; hold on;
for i = 1:size(ciclos_ref,1)
    i1 = ciclos_ref(i,1); i2 = ciclos_ref(i,2);
    fill([t(i1) t(i2) t(i2) t(i1)], [0 0 1 1], ...
         'c', 'FaceAlpha', 0.18, 'EdgeColor','none');
end
legend('Envolvente','Ciclos'); xlabel('Tiempo (s)'); ylabel('Amplitud norm.');
title('Ciclos cardiacos (propuestos)');
```

---

# BLOQUE 6 · Exportación de resultados para la **siguiente práctica**

### Teoría

Guardamos **índices** y **tiempos** $[t_{\text{ini}}, t_{\text{fin}}]$ de cada ciclo. Esto se usará después para **recortar ciclos** y extraer **características cepstrales** (MFCC/cepstrum clásico, etc.).

### Código

```matlab
ciclos_t = [ t(ciclos_ref(:,1)) , t(ciclos_ref(:,2)) ];
save resultados_segmentacion.mat fs t x Env ciclos_ref ciclos_t
disp('>> Segmentación completada. Guardado: resultados_segmentacion.mat');
```

---

## Experimentos (incluir en reporte)

1. Repite con **3 audios** (normal + patológicos).
2. Cambia $f_c$ del LPF (8, 10, 12 Hz). ¿Cómo afecta extremos y ciclos?
3. Cambia la **base del log** (ln vs log10). ¿Diferencias prácticas?
4. Agrega opcionalmente un **pasa-bandas 20–400 Hz** **antes** de Shannon si el audio es ruidoso.
5. Reporta: capturas (envolvente+extremos, triángulos, ciclos), **#ciclos**, **RR medio**, observaciones, falsos ± y cómo los mitigaste.

---

## Pasos que se realizarán en las siguientes prácticas:

* **Extracción de características por ciclo**: **MFCC/cepstrum** del PCG, energía por ventana, ZCR, etc.
* **Construcción de dataset** (características + etiquetas).
* **Entrenamiento ML** (SVM/kNN/RF) y métricas.
