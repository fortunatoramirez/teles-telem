# Práctica 5 — Segmentación y dataset de PCG + **MFCC** por ciclo

**Curso:** Telesalud y Telemedicina
**Secuencia:** (Prác. 4) Segmentación por Shannon → **(Prác. 5) Selección robusta + Dataset + MFCC por ciclo**

---

## Visión general (por qué este pipeline)

1. **Shannon** sobre el PCG realza transiciones energéticas breves (S1, S2) al penalizar amplitudes pequeñas y destacar picos:
   [
   E[n] ;=; -,p[n]\log\big(p[n]+\varepsilon\big),\quad p[n]=\frac{|x[n]|}{\max_k |x[k]|+\varepsilon} \in [0,1].
   ]
   Con un **LPF** (8–12 Hz) sobre (E[n]) obtenemos una **envolvente** estable de los latidos.

2. En la envolvente, **máximos** flanqueados por **mínimos** describen un evento acústico. El triplete mín–máx–mín define un **triángulo (tiempo, amplitud)** cuya **área** aproxima prominencia temporal-espectral del evento.
   [
   A = \tfrac12 \big| x_1(y_2-y_3)+x_2(y_3-y_1)+x_3(y_1-y_2)\big|.
   ]

3. Para segmentar **sin parámetros fisiológicos** (sin BPM, sin RR), usamos un **umbral adaptativo** de área:
   [
   A_{\text{thr}} ;=; 1.10,\bar A,
   ]
   que descarta automáticamente eventos pequeños/ruidosos. Luego resolvemos solapes con un **algoritmo greedy** ordenado por área (primero los eventos más “fuertes”).

4. Cada ciclo segmentado se convierte en una fila de **dataset**. Después calculamos **MFCC por ciclo** (promedio y desviación por coeficiente) para armar un **vector de rasgos fijo** por latido, listo para ML.

---

# BLOQUE 1 · Selección de triángulos y **ciclos**

*(Retoma el Bloque 5 de la Práctica 4. Aquí sustituimos la limpieza fisiológica por un método totalmente automático.)*

### 1.1 Teoría

**a) Por qué la “área” funciona como medida de prominencia**

* Un evento cardíaco audible (S1 o S2) se refleja en la envolvente como un **pulso** con “subida-pico-bajada”.
* El triángulo mín–máx–mín en el plano ((t,\text{Env})) resume **duración efectiva** (base) y **intensidad** (altura).
* Para un pulso (y(t)) con pico (A) y anchura aproximada (T), su “energía de envolvente” escala como (A\cdot T). El área del triángulo es proporcional a esa magnitud: mayor área ⇒ evento más **coherente** y **dominante**.

**b) Por qué el umbral (A_{\text{thr}} = 1.10,\bar A)**

* (\bar A) captura la **escala típica** de los eventos de ese archivo.
* Multiplicar por (1.10) (**+10 %**) añade **margen de seguridad** contra picos falsos y ruido leve; es un **offset relativo**, no absoluto, por lo que **se adapta** a cada señal.

**c) Por qué un greedy por área (y no por tiempo)**

* El problema es un **“interval scheduling” con pesos** (el peso es el área). La solución óptima exacta requiere DP (Weighted Interval Scheduling).
* Sin embargo, en PCG típicamente los eventos prominentes están **bien separados**, y un **greedy descendente en área** con descarte de solapes captura muy bien los latidos relevantes con **costo lineal-logarítmico** y **sin hiperparámetros** (ni RR, ni BPM).
* El orden por área **prioriza S1/S2 reales** sobre fluctuaciones residuales.

**d) No usar duración fisiológica**

* Evitamos rechazar latidos atípicos (taquicardia/bradicardia, soplos que alargan pulsos, registros pediátricos), manteniendo el algoritmo **agnóstico a la clínica** y **reproducible**.

### 1.2 Algoritmo (paso a paso, textual)

1. Calcular todas las áreas (A_i) de los triángulos mín–máx–mín detectados.
2. Calcular (\bar A) y fijar (A_{\text{thr}}=1.10,\bar A).
3. Conservar candidatos con (A_i \ge A_{\text{thr}}).
4. Ordenar candidatos **por área descendente**.
5. Recorrer en ese orden y **aceptar** un candidato si su intervalo ([t_{\min1}, t_{\min2}]) **no solapa** con uno ya aceptado; en caso de solape, **se ignora** (porque ya se aceptó uno mayor).
6. Ordenar los ciclos finales por tiempo de inicio y guardar ([i_{\min1}, i_{\min2}]) y ([t_{\min1}, t_{\min2}]).

### 1.3 Código

```matlab
%% === Entrada asumida desde Prác. 4: t, Env, tri_samp, tri_time, areas ===
Amean  = mean(areas);
Athr   = 1.10 * Amean;                 % umbral robusto (+10% sobre el promedio)
mask   = (areas >= Athr);

cand_samp = tri_samp(mask, :);         % [imin1, imax, imin2]
cand_time = tri_time(mask, :);         % [tmin1, tmax, tmin2]
cand_area = areas(mask);

% Greedy por área (desc), evitando solapes
[~, order] = sort(cand_area, 'descend');
chosen = [];        % filas: [t_ini, t_fin, area, idx_candidato]
used   = false(size(cand_area));

for k = 1:numel(order)
    r = order(k);
    ti = cand_time(r,1); tf = cand_time(r,3);

    % ¿solapa con algún aceptado?
    overlap = any( (ti < chosen(:,2)) & (tf > chosen(:,1)) ); % intersección real
    if ~overlap
        chosen = [chosen; ti, tf, cand_area(r), r]; %#ok<AGROW>
        used(r) = true;
    end
end

sel_rows     = find(used);
ciclos_idx   = cand_samp(sel_rows, [1 3]);             % [imin1, imin2]
ciclos_time  = cand_time(sel_rows, [1 3]);             % [tmin1, tmin2]
ciclos_area  = cand_area(sel_rows);

% Ordenar por tiempo de inicio
[~, ordt]    = sort(ciclos_time(:,1), 'ascend');
ciclos_idx   = ciclos_idx(ordt,:); 
ciclos_time  = ciclos_time(ordt,:);
ciclos_area  = ciclos_area(ordt);

% Visual
figure('Name','Ciclos cardiacos (umbral 1.10·Amean + greedy)');
plot(t, Env, 'k'); grid on; hold on;
for i = 1:size(ciclos_idx,1)
    i1 = ciclos_idx(i,1); i2 = ciclos_idx(i,2);
    fill([t(i1) t(i2) t(i2) t(i1)], [0 0 1 1], ...
         'c', 'FaceAlpha', 0.18, 'EdgeColor','none');
end
title('Envolvente de Shannon + ciclos seleccionados');
xlabel('Tiempo (s)'); ylabel('Amplitud norm.'); legend('Envolvente','Ciclos');
```

---

# BLOQUE 2 · **Dataset de ciclos** (Bloque 6 - Práctica anterior)

### 2.1 Teoría ampliada (por qué y cómo guardar)

* Un **dataset de ciclos** desacopla **segmentación** de **feature engineering/ML**.
* Debe registrar lo necesario para reproducir y auditar: **origen**, **tiempos**, **índices**, **prominencia (área)** y **fs**.
* Recomendación metodológica: **no mezclar** ciclos del **mismo paciente** entre train/test (evita “leakage”). Si el origen no está disponible, al menos consérvalo en `file_id` para split por archivo.

**Estructura propuesta (por fila = 1 ciclo):**
`file_id`, `cycle_id`, `i_start`, `i_end`, `t_start`, `t_end`, `n_samples`, `area`, `fs`.

### 2.2 Pasos

1. Inicializar tabla con nombres de columnas.
2. Para cada ciclo aceptado, llenar sus metadatos.
3. Guardar en **MAT** (rápido en MATLAB) y **CSV** (intercambio).
4. (Opcional) Añadir columnas `patient_id` o `source` si estuvieran disponibles.

### 2.3 Código

```matlab
[file_path, file_name, ~] = fileparts('data/07_Apex_MidSysMur_Supine_Bell.mp3'); % ajusta a tu ruta

nC = size(ciclos_idx,1);
file_id   = repmat(string(file_name), nC, 1);
cycle_id  = (1:nC).';

i_start   = ciclos_idx(:,1);
i_end     = ciclos_idx(:,2);
t_start   = ciclos_time(:,1);
t_end     = ciclos_time(:,2);
n_samples = i_end - i_start + 1;
area_sel  = ciclos_area(:);
fs_col    = repmat(fs, nC, 1);

T_cycles = table(file_id, cycle_id, i_start, i_end, t_start, t_end, n_samples, area_sel, fs_col, ...
   'VariableNames', {'file_id','cycle_id','i_start','i_end','t_start','t_end','n_samples','area','fs'});

if ~exist('out','dir'); mkdir out; end
save(fullfile('out', file_name + "_cycles.mat"), 'T_cycles');
writetable(T_cycles, fullfile('out', file_name + "_cycles.csv"));
disp(">> Guardado dataset base de ciclos: out/" + file_name + "_cycles.(mat,csv)");
```

---

# BLOQUE 3 · **MFCC por ciclo** (fundamento teórico)

**a) Mel y percepción**

* La escala **Mel** aproxima la **resolución frecuencial no lineal** del oído: más fina a bajas frecuencias y más gruesa a altas.
  Conversión típica:
  [
  m(f) = 2595,\log_{10}!\left(1+\frac{f}{700}\right),\qquad
  f(m) = 700\left(10^{m/2595}-1\right).
  ]

**b) STFT y banco de filtros Mel**

* Para cada ventana (x_m[n]) (Hamming, 20–30 ms) se calcula el **espectro de potencia** (P_m[k]).
* Se aplica un **banco triangular Mel** (H_j[k]) (típicamente (J\in[20,40])):
  [
  E_m(j) = \sum_k H_j[k],P_m[k].
  ]

**c) Log-energías y DCT (decorrelación)**

* Se toman **log-energías** (L_m(j)=\log(E_m(j)+\varepsilon)) para estabilizar variaciones multiplicativas.
* La **DCT-II** comprime y **decorrela**:
  [
  c_m(q)=\sum_{j=1}^{J} L_m(j)\cos!\Big[\tfrac{\pi}{J}q,(j-\tfrac12)\Big],;q=0,\dots,Q-1.
  ]
  Usamos (Q=13) típicamente (de (c_0) a (c_{12})).

**d) Agregación por ciclo (fijar dimensión)**

* Un ciclo tiene **duración variable**. Para obtener un **vector fijo**, agregamos por ventana:
  [
  \mu_q=\operatorname{mean}*m{c_m(q)},\qquad
  \sigma_q=\operatorname{std}*m{c_m(q)}.
  ]
  Vector final del ciclo: ([\mu_0,\dots,\mu*{Q-1},\sigma_0,\dots,\sigma*{Q-1}]).

**e) Detalles prácticos (robustez numérica)**

* **Pre-énfasis** (opcional) si la señal es muy atenuada en altas.
* **Floor** numérico (\varepsilon) antes del log para evitar (\log(0)).
* **Normalización** posterior (z-score por columna) se hace **después** de construir el dataset global (Prác. 6).

---

# BLOQUE 4 · Implementación MFCC (con fallback)

Incluye helpers para banco Mel si no tienes Audio Toolbox.

```matlab
%% Helpers: conversión Mel y banco de filtros
function m = hz2mel(hz), m = 2595*log10(1+hz/700); end
function hz = mel2hz(m), hz = 700*(10.^(m/2595)-1); end

function H = melFilterBank(n_mels, nfft, fs, fmin, fmax)
    if nargin < 4, fmin = 0; end
    if nargin < 5, fmax = fs/2; end
    mels   = linspace(hz2mel(fmin), hz2mel(fmax), n_mels+2);
    hz     = mel2hz(mels);
    bins   = floor( (nfft+1) * hz / fs );
    H      = zeros(n_mels, floor(nfft/2)+1);
    for m = 1:n_mels
        fL=bins(m); fC=bins(m+1); fR=bins(m+2);
        for k = fL:fC, if k>=0 && k<=floor(nfft/2), H(m,k+1) = (k-fL)/max(1,(fC-fL)); end, end
        for k = fC:fR, if k>=0 && k<=floor(nfft/2), H(m,k+1) = (fR-k)/max(1,(fR-fC)); end, end
    end
end
```

```matlab
%% MFCC por ciclo (usa built-in si existe; si no, fallback manual)
function feats = mfcc_per_cycle(xc, fs, pars)
    if nargin < 3
        pars.frame_ms = 25; pars.hop_ms = 10;
        pars.n_mels = 26;   pars.n_mfcc = 13;
    end

    % Framing
    frame_len = round(pars.frame_ms*1e-3*fs);
    hop_len   = round(pars.hop_ms*1e-3*fs);
    nfft      = 2^nextpow2(frame_len);
    win       = hamming(frame_len, 'periodic');

    if exist('mfcc','file') == 2
        [C, ~] = mfcc(xc, fs, 'WindowLength', frame_len, 'OverlapLength', frame_len-hop_len, ...
                           'NumCoeffs', pars.n_mfcc, 'LogEnergy', 'Ignore');
        mu  = mean(C, 1, 'omitnan');
        sig = std(C,  0, 1, 'omitnan');
        feats = [mu sig];
        return;
    end

    % Fallback manual
    H      = melFilterBank(pars.n_mels, nfft, fs, 0, fs/2);
    nF     = max(0, 1 + floor((length(xc)-frame_len)/hop_len));
    if nF==0
        % ciclo extremadamente corto: pad simple
        xc = [xc; zeros(max(0,frame_len-length(xc)),1)];
        nF = 1;
    end
    C = zeros(nF, pars.n_mfcc);

    idx = 1;
    for m = 1:nF
        seg = xc(idx:min(idx+frame_len-1, length(xc)));
        if length(seg)<frame_len, seg(end+1:frame_len)=0; end
        idx = idx + hop_len;

        X  = fft(seg.*win, nfft);
        P  = (abs(X(1:floor(nfft/2)+1)).^2) / frame_len;
        Em = H * P(:);
        Lm = log(Em + eps);

        D  = dctmtx(pars.n_mels);
        c  = D(1:pars.n_mfcc,:) * Lm(:);
        C(m,:) = c(:).';
    end

    mu  = mean(C, 1, 'omitnan');
    sig = std(C,  0, 1, 'omitnan');
    feats = [mu sig];
end
```

---

# BLOQUE 5 · Dataset de **rasgos MFCC** por ciclo + **etiquetas**

### 5.1 Teoría (etiquetado y trazabilidad)

* **Etiqueta por archivo** (simple y segura): todos los ciclos provenientes del mismo archivo comparten `label`.
* **Buenas prácticas para ML biomédico:**

  * Split **por paciente/origen**, nunca por muestras aleatorias (evita “leakage”).
  * Conservar `file_id` y, si se tiene, `patient_id` para particionar a futuro.
  * Mantener una **semilla** y registrar versiones del pipeline.

### 5.2 Código (con inferencia de etiqueta por nombre)

```matlab
function label = infer_label_from_filename(file_name)
    fn = lower(string(file_name));
    if contains(fn, "normal")
        label = "Normal";
    elseif contains(fn, "mur") || contains(fn,"murmur")
        label = "Murmur";
    elseif contains(fn, "click")
        label = "Click";
    else
        label = "Unknown";
    end
end
```

```matlab
%% Construcción del dataset de rasgos MFCC por ciclo
S = load(fullfile('out', file_name + "_cycles.mat"));   % T_cycles
T_cycles = S.T_cycles;

% Etiqueta (ajusta si quieres hacerlo manual)
target_label = infer_label_from_filename(file_name);

pars.frame_ms = 25; pars.hop_ms = 10; pars.n_mels = 26; pars.n_mfcc = 13;

feat_names_mu  = "mfcc" + (0:pars.n_mfcc-1) + "_mean";
feat_names_std = "mfcc" + (0:pars.n_mfcc-1) + "_std";
all_feat_names = [feat_names_mu, feat_names_std];

rows = height(T_cycles);
F = zeros(rows, numel(all_feat_names));

for r = 1:rows
    i1 = T_cycles.i_start(r);
    i2 = T_cycles.i_end(r);
    xc = x(i1:i2);                            % usa el mismo segmento normalizado
    feats = mfcc_per_cycle(xc, fs, pars);     % 2*n_mfcc rasgos
    F(r,:) = feats;
end

Label = repmat(string(target_label), rows, 1);

T_features = [ T_cycles(:, {'file_id','cycle_id','t_start','t_end','area'}) , ...
               array2table(F, 'VariableNames', all_feat_names), ...
               table(Label, 'VariableNames', {'label'}) ];

save(fullfile('out', file_name + "_mfcc_dataset.mat"), 'T_features', 'pars');
writetable(T_features, fullfile('out', file_name + "_mfcc_dataset.csv"));
disp(">> Guardado dataset de rasgos: out/" + file_name + "_mfcc_dataset.(mat,csv)");
```

> **Nota:** al trabajar con el **audio completo** (no sólo la ventana de 2 s), reconstruye cada ciclo desde el archivo original con sus **tiempos absolutos** ([t_{\text{ini}},t_{\text{fin}}]) para evitar inconsistencias.

---

# BLOQUE 6 · Procesamiento **batch** (varios archivos) y dataset global

**Objetivo:** concatenar filas de múltiples audios para tener un **dataset global** (`ALL_pcg_mfcc_dataset`).

**Buenas prácticas adicionales**

* Guardar `source_path`, `file_id`, `fs` en cada lote.
* Mantener **log** de parámetros: (f_c) del LPF, versión de Shannon, (A_{\text{thr}}), ventana/hop, (n_mels), (n_mfcc).

```matlab
files = dir(fullfile('data','*.mp3'));
All = table();

for f = 1:numel(files)
    this_file = fullfile(files(f).folder, files(f).name);
    % (1) Reproducir Bloques de la Prác. 4 para x,t,Env,triangulación
    % (2) Aplicar BLOQUE 1 (umbral 1.10·Amean + greedy) -> ciclos_idx/time
    % (3) BLOQUE 2 -> T_cycles
    % (4) BLOQUE 5 -> T_features
    S = load(fullfile('out', erase(files(f).name, ".mp3") + "_mfcc_dataset.mat"));
    All = [All; S.T_features]; %#ok<AGROW>
end

writetable(All, fullfile('out','ALL_pcg_mfcc_dataset.csv'));
save(fullfile('out','ALL_pcg_mfcc_dataset.mat'), 'All');
```

---
