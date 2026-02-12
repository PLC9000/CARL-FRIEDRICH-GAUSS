from app.strategies.strategy_a_arima import run_strategy_a
from app.strategies.strategy_b_kalman import run_strategy_b
from app.strategies.strategy_c_ou_mean_reversion import run_strategy_c
from app.strategies.strategy_d_ma_crossover import run_strategy_d
from app.strategies.strategy_e_rsi import run_strategy_e
from app.strategies.strategy_f_macd import run_strategy_f
from app.strategies.strategy_g_bollinger import run_strategy_g
from app.strategies.strategy_h_breakout import run_strategy_h
from app.strategies.strategy_i_atr_trailing import run_strategy_i
from app.strategies.strategy_j_adx import run_strategy_j
from app.strategies.strategy_k_fng import run_strategy_k
from app.strategies.strategy_l_ema_slope import run_strategy_l

STRATEGY_MAP = {
    "A": run_strategy_a,
    "B": run_strategy_b,
    "C": run_strategy_c,
    "D": run_strategy_d,
    "E": run_strategy_e,
    "F": run_strategy_f,
    "G": run_strategy_g,
    "H": run_strategy_h,
    "I": run_strategy_i,
    "J": run_strategy_j,
    "K": run_strategy_k,
    "L": run_strategy_l,
}

# ── Strategy registry metadata ───────────────────────────────────────
# Used by the /strategies endpoints and the frontend to display info.

STRATEGY_REGISTRY = {
    "A": {
        "key": "A",
        "name": "Prediccion (AR)",
        "icon": "\U0001f4c8",
        "category": "Modelos Predictivos",
        "short_desc": "Mira los ultimos precios para predecir hacia donde ira el proximo.",
        "long_desc": (
            "Imaginate que el precio de una moneda es como el clima: si los ultimos "
            "dias fueron calurosos, es probable que manana tambien lo sea. Esta "
            "estrategia hace exactamente eso con los precios. Analiza el patron de "
            "los ultimos periodos (por defecto 4) y calcula matematicamente hacia "
            "donde es mas probable que se mueva el precio en el futuro cercano. "
            "Si la prediccion dice que el precio va a subir mas de un cierto %, "
            "recomienda COMPRAR. Si dice que va a bajar, recomienda VENDER. "
            "Si no esta seguro, dice ESPERAR."
        ),
        "when_works": "Cuando el mercado tiene una direccion clara: subiendo o bajando de forma sostenida. Funciona bien con BTC y ETH en momentos de tendencia.",
        "when_fails": "Cuando el precio se mueve de costado sin direccion clara, o cuando hay noticias imprevistas que cambian todo de golpe (ej: un hackeo, regulacion nueva).",
        "example": "Si BTC subio 2% ayer, 1.5% anteayer y 1% hace 3 dias, la estrategia predice que manana seguira subiendo y recomienda COMPRAR.",
        "default_params": {
            "ar_order": {
                "value": 4, "min": 2, "max": 10,
                "desc": "Cuantas velas pasadas usa para hacer la prediccion.",
                "tip": "Valor bajo (2-3) = reacciona rapido pero menos preciso. Valor alto (7-10) = mas preciso pero mas lento para detectar cambios.",
            },
            "forecast_horizon": {
                "value": 5, "min": 1, "max": 20,
                "desc": "Cuantas velas hacia adelante intenta predecir.",
                "tip": "Con 1-3 se enfoca en el corto plazo (scalping). Con 10-20 busca movimientos mas grandes.",
            },
            "return_threshold": {
                "value": 0.5, "min": 0.1, "max": 5.0, "step": 0.1,
                "desc": "Porcentaje minimo de movimiento predicho para generar senal.",
                "tip": "Si lo pones en 0.1% genera muchas senales (mas operaciones). En 2%+ solo genera senal con movimientos grandes (menos operaciones, mas seguras).",
            },
            "sl_multiplier": {
                "value": 1.5, "min": 0.5, "max": 5.0, "step": 0.1,
                "desc": "Que tan lejos poner el stop-loss para limitar perdidas.",
                "tip": "Se calcula como multiplo de la volatilidad reciente (ATR). Con 1.0 el stop queda apretado (te saca rapido). Con 3.0+ le da mas espacio al precio para moverse.",
            },
            "tp_multiplier": {
                "value": 2.5, "min": 0.5, "max": 10.0, "step": 0.1,
                "desc": "Que tan lejos poner el take-profit para tomar ganancias.",
                "tip": "Idealmente debe ser mayor que sl_multiplier. Ej: SL=1.5 y TP=2.5 significa que por cada $1 que arriesgas, buscas ganar $1.67.",
            },
        },
    },
    "B": {
        "key": "B",
        "name": "Tendencia (Kalman)",
        "icon": "\U0001f52c",
        "category": "Modelos Predictivos",
        "short_desc": "Filtra el ruido del precio para encontrar la tendencia real oculta.",
        "long_desc": (
            "El precio de una crypto se mueve todo el tiempo con pequenos saltos "
            "aleatorios (ruido), lo que dificulta ver si realmente esta subiendo o "
            "bajando. Esta estrategia usa un filtro matematico avanzado (Kalman) para "
            "separar el ruido del movimiento real. Es como ponerse lentes que te "
            "dejan ver solo la tendencia verdadera. Si la tendencia real apunta hacia "
            "arriba, recomienda COMPRAR. Si apunta abajo, VENDER. Ademas, si detecta "
            "que el mercado esta demasiado volatil (caos), se calla y dice ESPERAR "
            "porque en esos momentos nadie puede predecir nada."
        ),
        "when_works": "Cuando hay una tendencia sostenida pero con mucho ruido en el camino. Ideal para timeframes de 1h-4h donde hay tendencia pero el minuto a minuto confunde.",
        "when_fails": "En mercados extremadamente volatiles (ej: cuando sale una noticia bomba) o cuando el precio se mueve en un rango sin direccion durante mucho tiempo.",
        "example": "BTC esta en 70.000 con subas y bajas constantes de $200-$500. El filtro detecta que detras de todo ese ruido, la tendencia real sube $50/hora y recomienda COMPRAR.",
        "default_params": {
            "process_noise": {
                "value": 0.00001, "min": 0.000001, "max": 0.001, "step": 0.000001,
                "desc": "Que tan rapido se adapta el filtro a cambios reales en la tendencia.",
                "tip": "Valor bajo = filtro suave que tarda en reaccionar pero da menos senales falsas. Valor alto = reacciona rapido pero puede confundir ruido con tendencia. Dejalo en el default salvo que sepas lo que haces.",
            },
            "measurement_noise": {
                "value": 0.01, "min": 0.001, "max": 1.0, "step": 0.001,
                "desc": "Cuanto 'ruido' asume que tiene cada precio que ve.",
                "tip": "Valor alto = el filtro 'desconfia' mas de cada precio individual y suaviza mas. Util si operas en 1m donde hay mucho ruido. Para 1h-4h el default funciona bien.",
            },
            "trend_threshold": {
                "value": 0.03, "min": 0.005, "max": 0.5, "step": 0.005,
                "desc": "Pendiente minima de la tendencia (%) para generar senal.",
                "tip": "Con 0.01% genera senal con cualquier movimiento minimo. Con 0.1%+ solo reacciona a tendencias claras. Empeza con el default.",
            },
            "volatility_cap": {
                "value": 5.0, "min": 1.0, "max": 20.0, "step": 0.5,
                "desc": "Si la volatilidad supera este %, la estrategia se silencia y dice ESPERAR.",
                "tip": "Es un filtro de seguridad. Con 3% se calla con cualquier movimiento fuerte. Con 10%+ solo se calla en crashes extremos. El default de 5% es un buen balance.",
            },
            "sl_multiplier": {
                "value": 1.5, "min": 0.5, "max": 5.0, "step": 0.1,
                "desc": "Distancia del stop-loss como multiplo de la volatilidad (ATR).",
                "tip": "Con 1.0 el stop queda cerca (te saca rapido si se da vuelta). Con 3.0+ le da espacio para respirar.",
            },
            "tp_multiplier": {
                "value": 2.5, "min": 0.5, "max": 10.0, "step": 0.1,
                "desc": "Distancia del take-profit como multiplo de la volatilidad (ATR).",
                "tip": "Un TP mayor que el SL hace que ganes mas de lo que arriesgas en cada operacion. Ratio 1:2 (SL=1.5, TP=2.5) es un clasico.",
            },
        },
    },
    "C": {
        "key": "C",
        "name": "Reversion a la Media (OU)",
        "icon": "\U0001f504",
        "category": "Contrarian",
        "short_desc": "Compra cuando el precio cayo demasiado, vende cuando subio demasiado.",
        "long_desc": (
            "Esta estrategia se basa en una idea simple: el precio tiende a volver "
            "a su promedio. Si BTC normalmente cotiza alrededor de $70.000 y de "
            "repente cae a $67.000 sin razon fundamental, es probable que vuelva "
            "a subir. La estrategia mide que tan lejos esta el precio de su promedio "
            "historico usando un puntaje llamado 'z-score'. Si esta muy por debajo "
            "(z-score negativo extremo) recomienda COMPRAR porque esta 'barato'. "
            "Si esta muy por arriba (z-score positivo extremo) recomienda VENDER. "
            "Es como ir al supermercado y comprar solo lo que esta en oferta."
        ),
        "when_works": "Mercados que se mueven en un rango predecible (ej: altcoins estables en periodos tranquilos). Excelente cuando no hay noticias fuertes y el precio oscila naturalmente.",
        "when_fails": "Cuando arranca una tendencia fuerte. Si BTC empieza a subir de $70K a $80K, esta estrategia va a decir VENDER a los $73K pensando que esta 'caro' y se pierde toda la subida.",
        "example": "ETH cotiza entre $3.300 y $3.500 hace una semana. De repente cae a $3.250. La estrategia detecta que esta 2 desviaciones por debajo del promedio y recomienda COMPRAR esperando que vuelva a $3.400.",
        "default_params": {
            "lookback": {
                "value": 50, "min": 20, "max": 200,
                "desc": "Cuantas velas pasadas usa para calcular el precio 'normal' (promedio).",
                "tip": "Con 20-30 reacciona rapido a cambios recientes. Con 100-200 usa un promedio mas largo y estable. Para 1h usa 50-100. Para 1m usa 20-50.",
            },
            "z_threshold": {
                "value": 2.0, "min": 1.0, "max": 4.0, "step": 0.1,
                "desc": "Que tan lejos del promedio debe estar el precio para generar senal.",
                "tip": "Con 1.5 genera senales frecuentes (muchas oportunidades, mas riesgo). Con 3.0+ solo senales en extremos muy raros (pocas operaciones, mas seguras). El clasico es 2.0.",
            },
            "vol_regime_multiplier": {
                "value": 1.5, "min": 1.0, "max": 3.0, "step": 0.1,
                "desc": "Filtro de seguridad: si la volatilidad actual es X veces mayor que lo normal, se silencia.",
                "tip": "Evita operar en momentos de panico o euforia donde el precio puede seguir alejandose del promedio. Con 1.2 es muy conservador. Con 2.5+ deja operar en casi cualquier condicion.",
            },
            "sl_multiplier": {
                "value": 1.5, "min": 0.5, "max": 5.0, "step": 0.1,
                "desc": "Distancia del stop-loss como multiplo de la volatilidad.",
                "tip": "Para reversion a la media, un SL ajustado (1.0-1.5) funciona bien porque esperas que el precio se de vuelta rapido.",
            },
            "tp_multiplier": {
                "value": 2.0, "min": 0.5, "max": 10.0, "step": 0.1,
                "desc": "Distancia del take-profit como multiplo de la volatilidad.",
                "tip": "El target natural es el promedio del precio. No pongas TPs demasiado ambiciosos en esta estrategia (2.0-3.0 es ideal).",
            },
        },
    },
    "D": {
        "key": "D",
        "name": "EMA Crossover",
        "icon": "\u2702\ufe0f",
        "category": "Contrarian / Reversion",
        "short_desc": "Score continuo basado en distancia entre EMAs, normalizado por volatilidad.",
        "long_desc": (
            "Calcula la separacion relativa entre una EMA rapida y una lenta, la "
            "normaliza por la volatilidad reciente (std de retornos) y la mapea a un "
            "score de -100 a +100 usando la funcion tanh. Cuando la EMA rapida esta "
            "MUY por encima de la lenta (el precio subio demasiado rapido), el score "
            "es positivo y recomienda VENDER (sobreextension). Cuando esta MUY por "
            "debajo (el precio cayo demasiado), el score es negativo y recomienda "
            "COMPRAR (depresion). Si la diferencia es pequena (zona muerta), no opera. "
            "Al normalizar por volatilidad, el score es comparable entre distintos "
            "tokens: un diff del 0.5% en BTC (baja vol) tiene el mismo peso que un "
            "diff del 3% en una altcoin (alta vol)."
        ),
        "when_works": "Cuando el precio se aleja temporalmente de su tendencia y luego revierte. Excelente para detectar sobreextensiones en pares con volatilidad regular. Funciona mejor en timeframes de 1h-4h.",
        "when_fails": "En tendencias parabolicas donde el precio sigue alejandose de la EMA lenta sin revertir. Tambien falla si la volatilidad cambia bruscamente (ej: flash crash) porque la normalizacion se distorsiona.",
        "example": "BTC sube rapido de $95K a $100K. La EMA(9) sube a $99.500 mientras la EMA(21) recien llega a $97.000. Diff normalizado da +65% → VENDER. Dos dias despues BTC corrige a $97.500.",
        "default_params": {
            "ema_fast_period": {
                "value": 9, "min": 3, "max": 50,
                "desc": "Periodos de la EMA rapida.",
                "tip": "Combinaciones clasicas: 9/21 (intraday), 12/26 (MACD-like), 20/50 (swing). Siempre menor que ema_slow_period.",
            },
            "ema_slow_period": {
                "value": 21, "min": 10, "max": 200,
                "desc": "Periodos de la EMA lenta (referencia estable).",
                "tip": "Mayor diferencia con la rapida = detecta desviaciones mas grandes. 21 es el clasico intraday, 50 para swing.",
            },
            "volatility_period": {
                "value": 14, "min": 5, "max": 50,
                "desc": "Cuantas velas usa para calcular la volatilidad (std de retornos).",
                "tip": "14 es el estandar. Con 7 reacciona mas rapido a cambios de volatilidad. Con 30+ suaviza mas.",
            },
            "k": {
                "value": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                "desc": "Sensibilidad del tanh. Controla que tan rapido el score satura a +-100%.",
                "tip": "Con 0.5 el score es mas suave (necesita divergencias grandes para llegar a +-80%). Con 2.0+ satura rapido (cualquier divergencia da scores extremos). 1.0 es un buen balance.",
            },
            "deadzone": {
                "value": 3.0, "min": 0.0, "max": 20.0, "step": 0.5,
                "desc": "Porcentaje minimo de |score| para generar senal. Por debajo = HOLD.",
                "tip": "Con 0 genera senal siempre (incluso con divergencias minimas). Con 5-10 filtra ruido y solo opera con senales claras. 3.0 es un buen default.",
            },
            "sl_multiplier": {
                "value": 1.5, "min": 0.5, "max": 5.0, "step": 0.1,
                "desc": "Distancia del stop-loss como multiplo del ATR.",
                "tip": "Para reversion a la media, 1.5-2.0 es razonable. Mas ajustado si el score es alto (alta confianza).",
            },
            "tp_multiplier": {
                "value": 2.5, "min": 0.5, "max": 10.0, "step": 0.1,
                "desc": "Distancia del take-profit como multiplo del ATR.",
                "tip": "El target natural es cuando las EMAs convergen. Un TP de 2.0-3.0 es coherente.",
            },
        },
    },
    "E": {
        "key": "E",
        "name": "RSI (Fuerza Relativa)",
        "icon": "\U0001f4ca",
        "category": "Osciladores",
        "short_desc": "Detecta cuando el mercado exagero: compra en panico, vende en euforia.",
        "long_desc": (
            "El RSI es como un termometro del mercado que va de 0 a 100. Mide si "
            "los compradores o los vendedores tienen mas fuerza en este momento. "
            "Cuando el RSI baja de 30, significa que los vendedores exageraron y el "
            "precio podria rebotar (momento de COMPRAR). Cuando sube de 70, los "
            "compradores exageraron y el precio podria caer (momento de VENDER). "
            "Entre 30 y 70 no hay senal clara. Es uno de los indicadores mas populares "
            "y confiables del analisis tecnico, creado por J. Welles Wilder en 1978."
        ),
        "when_works": "Ideal cuando el mercado oscila en un rango. Si BTC se mueve entre $68K y $72K, el RSI detecta muy bien los extremos para comprar abajo y vender arriba.",
        "when_fails": "En tendencias muy fuertes. Si BTC pasa de $60K a $80K en una semana, el RSI va a decir 'sobrecomprado' a los $65K y te perderas toda la subida. El RSI puede quedarse en 80+ durante tendencias alcistas fuertes.",
        "example": "BTC cae de $71.000 a $68.500 en pocas horas. El RSI baja a 25 (zona de sobreventa/panico). La estrategia recomienda COMPRAR. Dos horas despues, BTC rebota a $70.000.",
        "default_params": {
            "rsi_period": {
                "value": 14, "min": 5, "max": 50,
                "desc": "Cuantas velas usa para calcular el RSI.",
                "tip": "14 es el valor clasico y el mas testeado. Con 7 reacciona mas rapido (mas senales, mas falsas). Con 21+ es mas suave (menos senales, mas fiables). Para crypto volatil, 10-14 es ideal.",
            },
            "overbought": {
                "value": 70, "min": 60, "max": 90,
                "desc": "Por encima de este nivel se considera 'sobrecomprado' y genera senal de VENTA.",
                "tip": "70 es el clasico. Si queres ser mas conservador usa 80 (solo vende en euforia extrema). En mercados alcistas algunos usan 80 como umbral porque el RSI tiende a estar mas alto.",
            },
            "oversold": {
                "value": 30, "min": 10, "max": 40,
                "desc": "Por debajo de este nivel se considera 'sobrevendido' y genera senal de COMPRA.",
                "tip": "30 es el clasico. Si queres ser mas conservador usa 20 (solo compra en panico real). Para crypto, 25-30 funciona bien porque las caidas suelen ser bruscas.",
            },
            "sl_multiplier": {
                "value": 1.5, "min": 0.5, "max": 5.0, "step": 0.1,
                "desc": "Distancia del stop-loss como multiplo de la volatilidad.",
                "tip": "El RSI opera en reversos, asi que el SL debe proteger si el precio sigue cayendo en vez de rebotar. Un SL de 1.5-2.0 ATR es razonable.",
            },
            "tp_multiplier": {
                "value": 2.5, "min": 0.5, "max": 10.0, "step": 0.1,
                "desc": "Distancia del take-profit como multiplo de la volatilidad.",
                "tip": "El target natural es cuando el RSI vuelve a la zona media (50). No seas demasiado ambicioso, 2.0-3.0 es un buen rango.",
            },
        },
    },
    "F": {
        "key": "F",
        "name": "MACD (Momentum)",
        "icon": "\U0001f4c9",
        "category": "Seguimiento de Tendencia",
        "short_desc": "Detecta cuando una tendencia esta ganando o perdiendo fuerza.",
        "long_desc": (
            "El MACD combina lo mejor de dos mundos: detecta tendencias Y mide su "
            "fuerza (momentum). Funciona con dos lineas: la linea MACD (diferencia "
            "entre dos promedios del precio) y la linea de Senal (un promedio del MACD "
            "mismo). Cuando la linea MACD cruza por encima de la Senal, el momentum "
            "es alcista: COMPRAR. Cuando cruza por debajo, es bajista: VENDER. "
            "Tambien tiene un histograma (barras) que muestra la distancia entre "
            "ambas lineas: barras crecientes = la tendencia se fortalece, barras "
            "decrecientes = la tendencia se debilita. Es como tener un velocimetro "
            "del precio."
        ),
        "when_works": "Excelente para detectar cambios de tendencia y confirmar que un movimiento tiene fuerza real. Funciona muy bien despues de un periodo lateral cuando el precio 'elige direccion'.",
        "when_fails": "En mercados laterales sin tendencia genera senales erraticas (compra/vende sin parar). Tambien da senales tardias: cuando el MACD confirma, a veces ya paso buena parte del movimiento.",
        "example": "BTC estuvo lateral 3 dias. La linea MACD cruza la Senal hacia arriba. La estrategia recomienda COMPRAR. BTC sube 4% en las siguientes 12 horas mientras el histograma crece.",
        "default_params": {
            "fast": {
                "value": 12, "min": 5, "max": 30,
                "desc": "Periodos de la EMA rapida (componente principal del MACD).",
                "tip": "12 es el clasico de Gerald Appel. Con 8 es mas sensible (reacciona antes pero mas ruido). No lo cambies mucho salvo que tengas experiencia.",
            },
            "slow": {
                "value": 26, "min": 15, "max": 60,
                "desc": "Periodos de la EMA lenta (referencia estable del MACD).",
                "tip": "26 es el valor estandar. La diferencia entre fast y slow determina la sensibilidad. Los valores clasicos 12/26 funcionan bien para la mayoria de timeframes.",
            },
            "signal": {
                "value": 9, "min": 3, "max": 20,
                "desc": "Periodos de la linea de Senal (promedio del MACD).",
                "tip": "9 es el estandar. Con 5 reacciona mas rapido (util para scalping). Con 14+ suaviza mas y da menos senales falsas.",
            },
            "trigger": {
                "value": "cross", "options": ["cross", "histogram"],
                "desc": "Que evento genera la senal de trading.",
                "tip": "'cross' = genera senal cuando las lineas se cruzan (clasico, mas claro). 'histogram' = genera senal cuando las barras cambian de signo (mas rapido, detecta cambios antes). Empeza con 'cross'.",
            },
            "sl_multiplier": {
                "value": 1.5, "min": 0.5, "max": 5.0, "step": 0.1,
                "desc": "Distancia del stop-loss como multiplo de la volatilidad.",
                "tip": "El MACD es una estrategia de tendencia, dale espacio al precio (1.5-2.5 es un buen rango).",
            },
            "tp_multiplier": {
                "value": 2.5, "min": 0.5, "max": 10.0, "step": 0.1,
                "desc": "Distancia del take-profit como multiplo de la volatilidad.",
                "tip": "Podes ser generoso (3.0-5.0) si el momentum es fuerte. El cruce inverso del MACD naturalmente sugiere cerrar.",
            },
        },
    },
    "G": {
        "key": "G",
        "name": "Bandas de Bollinger",
        "icon": "\U0001f30a",
        "category": "Volatilidad",
        "short_desc": "Bandas que se expanden y contraen con la volatilidad. Compra abajo, vende arriba.",
        "long_desc": (
            "Imaginate dos paredes invisibles alrededor del precio: una arriba y una "
            "abajo. Estas 'bandas' se calculan usando la volatilidad reciente. Cuando "
            "el mercado esta tranquilo, las bandas se acercan. Cuando esta loco, se "
            "alejan. La estrategia dice: si el precio toca la banda inferior, esta "
            "excesivamente barato para las condiciones actuales, COMPRAR. Si toca la "
            "superior, esta excesivamente caro, VENDER. Estadisticamente, el precio "
            "se mantiene dentro de las bandas el ~95% del tiempo (con 2 desviaciones "
            "estandar), asi que cuando sale es una senal significativa."
        ),
        "when_works": "Mercados que oscilan con volatilidad 'normal'. Ideal para pares estables como BTC/USDT en periodos tranquilos, o para altcoins que se mueven en un rango predecible.",
        "when_fails": "Cuando el precio rompe las bandas y sigue de largo (breakout). En una corrida alcista fuerte, el precio puede 'caminar' pegado a la banda superior sin volver al centro, dandote senales de venta que te sacan demasiado temprano.",
        "example": "ETH oscila entre $3.200 y $3.600. Las bandas de Bollinger estan en $3.180 (inferior) y $3.620 (superior). ETH baja a $3.180, la estrategia recomienda COMPRAR. ETH rebota a $3.400.",
        "default_params": {
            "period": {
                "value": 20, "min": 10, "max": 50,
                "desc": "Cuantas velas usa para calcular el centro de las bandas (media movil).",
                "tip": "20 es el clasico de John Bollinger. Con 10 las bandas reaccionan mas rapido (util para 1m-5m). Con 50 son mas estables (mejor para 4h-1d).",
            },
            "std_dev": {
                "value": 2.0, "min": 1.0, "max": 4.0, "step": 0.1,
                "desc": "Cuantas desviaciones estandar de distancia tienen las bandas del centro.",
                "tip": "Con 2.0 el precio se mantiene dentro el ~95% del tiempo (clasico). Con 1.5 las bandas estan mas cerca (mas senales, menos fiables). Con 2.5-3.0 solo genera senal en extremos muy raros.",
            },
            "entry_rule": {
                "value": "touch", "options": ["touch", "close_outside"],
                "desc": "Cuando se activa la senal: al tocar la banda o al cerrar fuera de ella.",
                "tip": "'touch' = genera senal apenas el precio roza la banda (mas rapido, mas senales). 'close_outside' = espera a que la vela cierre fuera de la banda (mas confirmacion, menos falsas alarmas).",
            },
            "sl_multiplier": {
                "value": 1.5, "min": 0.5, "max": 5.0, "step": 0.1,
                "desc": "Distancia del stop-loss como multiplo de la volatilidad.",
                "tip": "Como Bollinger ya mide volatilidad, un SL de 1.0-1.5 ATR suele ser suficiente.",
            },
            "tp_multiplier": {
                "value": 2.5, "min": 0.5, "max": 10.0, "step": 0.1,
                "desc": "Distancia del take-profit como multiplo de la volatilidad.",
                "tip": "El target natural es la banda opuesta o la media central. Un TP de 2.0-3.0 es coherente con el rango de las bandas.",
            },
        },
    },
    "H": {
        "key": "H",
        "name": "Ruptura de Canal (Breakout)",
        "icon": "\U0001f680",
        "category": "Seguimiento de Tendencia",
        "short_desc": "Detecta cuando el precio escapa de un rango: compra en nuevos maximos, vende en nuevos minimos.",
        "long_desc": (
            "A veces el precio se queda 'atrapado' en un rango durante dias, rebotando "
            "entre un maximo y un minimo sin decidirse. Cuando finalmente rompe ese techo "
            "o ese piso, suele arrancar un movimiento fuerte en esa direccion. Esta "
            "estrategia dibuja un 'canal' con el maximo mas alto y el minimo mas bajo "
            "de los ultimos N periodos. Si el precio rompe por arriba del canal, "
            "recomienda COMPRAR (nueva tendencia alcista). Si rompe por abajo, VENDER. "
            "Opcionalmente filtra por volumen: una ruptura con mucho volumen es mas "
            "confiable que una con poco."
        ),
        "when_works": "Justo despues de periodos de consolidacion (el precio lateral se 'comprime' y luego explota). Ideal para detectar el inicio de movimientos grandes. Funciona muy bien con BTC antes de rallies importantes.",
        "when_fails": "En mercados laterales con 'falsas rupturas': el precio pasa el canal por unos minutos y luego vuelve adentro, generando una senal falsa. Por eso existe el parametro de buffer.",
        "example": "BTC oscila entre $68.000 y $70.000 durante 5 dias. El canal superior esta en $70.000. BTC rompe a $70.200 con volumen alto. La estrategia recomienda COMPRAR. BTC sube a $73.000 en los siguientes 2 dias.",
        "default_params": {
            "lookback": {
                "value": 20, "min": 10, "max": 55,
                "desc": "Cuantas velas atras mira para definir el techo y piso del canal.",
                "tip": "20 es el clasico del Canal Donchian (usado por los famosos 'Turtle Traders'). Con 10 el canal es mas estrecho (detecta rupturas rapido). Con 55 (otro valor famoso) solo detecta rupturas de canales grandes.",
            },
            "buffer_pct": {
                "value": 0.1, "min": 0.0, "max": 1.0, "step": 0.05,
                "desc": "Porcentaje extra que el precio debe superar el canal para confirmar la ruptura.",
                "tip": "Con 0% cualquier tick que toque el canal genera senal (muchas falsas). Con 0.1-0.2% filtra la mayoria de las falsas rupturas. Con 0.5%+ solo senales muy obvias.",
            },
            "volume_filter": {
                "value": False,
                "desc": "Si se activa, solo genera senal cuando el volumen de la vela de ruptura es mayor al promedio.",
                "tip": "MUY recomendado activarlo. Una ruptura con volumen alto es mucho mas confiable. Sin volumen, muchas veces el precio vuelve al canal rapidamente.",
            },
            "sl_multiplier": {
                "value": 1.5, "min": 0.5, "max": 5.0, "step": 0.1,
                "desc": "Distancia del stop-loss como multiplo de la volatilidad.",
                "tip": "En breakouts el SL natural es justo debajo/encima del canal roto. Un multiplo de 1.0-1.5 ATR suele quedar bien alineado con ese nivel.",
            },
            "tp_multiplier": {
                "value": 2.5, "min": 0.5, "max": 10.0, "step": 0.1,
                "desc": "Distancia del take-profit como multiplo de la volatilidad.",
                "tip": "Los breakouts pueden generar movimientos MUY grandes. Podes ser ambicioso con 3.0-5.0 o incluso mas si el canal era muy largo (mucha energia acumulada).",
            },
        },
    },
    "I": {
        "key": "I",
        "name": "ATR Trailing Stop",
        "icon": "\U0001f6e1\ufe0f",
        "category": "Gestion de Riesgo",
        "short_desc": "Un stop-loss inteligente que se mueve con el precio y detecta cambios de tendencia.",
        "long_desc": (
            "En vez de poner un stop-loss fijo, esta estrategia usa uno que se mueve "
            "automaticamente con el precio. Si BTC sube, el stop sube tambien (pero "
            "nunca baja). Si BTC baja, el stop baja tambien (pero nunca sube). "
            "La distancia del stop al precio se calcula con el ATR (Average True Range), "
            "que mide la volatilidad real del mercado. Cuando el precio cruza el "
            "trailing stop hacia abajo, cambia de tendencia alcista a bajista (senal de "
            "VENTA). Cuando cruza hacia arriba, cambia a alcista (COMPRA). Es ideal "
            "como complemento de otras estrategias porque automaticamente protege "
            "ganancias y limita perdidas."
        ),
        "when_works": "Tendencias sostenidas donde queres quedarte en la operacion el mayor tiempo posible sin que un retroceso te saque. Excelente para 'dejar correr las ganancias'.",
        "when_fails": "Mercados muy erraticos que cambian de direccion constantemente. El trailing stop se activa una y otra vez generando operaciones perdedoras por los spreads y comisiones.",
        "example": "BTC sube de $68.000 a $72.000. El trailing stop empieza en $67.000 y va subiendo: $67.500, $68.200, $69.000, $70.500. Si BTC cae a $70.500 se activa el stop y recomienda VENDER, protegiendo gran parte de la ganancia.",
        "default_params": {
            "atr_period": {
                "value": 14, "min": 5, "max": 50,
                "desc": "Cuantas velas usa para calcular la volatilidad promedio (ATR).",
                "tip": "14 es el clasico. Con 7 el ATR reacciona rapido a cambios de volatilidad (stop mas dinamico). Con 21+ el ATR es mas estable (stop mas predecible).",
            },
            "atr_multiplier": {
                "value": 3.0, "min": 1.0, "max": 6.0, "step": 0.5,
                "desc": "Cuantas veces el ATR de distancia entre el precio y el trailing stop.",
                "tip": "Con 1.5-2.0 el stop queda cerca del precio (protege rapido, pero te saca con cualquier retroceso normal). Con 3.0 (default) aguanta retrocesos tipicos. Con 4.0-5.0 solo te saca en reversos serios. El famoso 'SuperTrend' usa 3.0.",
            },
            "direction": {
                "value": "both", "options": ["both", "long_only", "short_only"],
                "desc": "En que direccion opera la estrategia.",
                "tip": "'both' = compra y vende (default). 'long_only' = solo genera senales de COMPRA (ideal si sos optimista con el activo). 'short_only' = solo VENTA (raro en crypto, pero util para cobertura).",
            },
            "sl_multiplier": {
                "value": 1.0, "min": 0.5, "max": 5.0, "step": 0.1,
                "desc": "Stop-loss adicional de seguridad por debajo del trailing stop.",
                "tip": "Como la estrategia YA tiene un trailing stop dinamico, este SL adicional es un respaldo. Con 1.0 queda justo en el trailing. Con 2.0+ da un colchon extra por si hay un spike rapido.",
            },
            "tp_multiplier": {
                "value": 2.0, "min": 0.5, "max": 10.0, "step": 0.1,
                "desc": "Take-profit como multiplo de la volatilidad.",
                "tip": "Esta estrategia normalmente deja que el trailing stop maneje la salida. El TP es un limite maximo opcional. Podes ponerlo alto (5.0+) para dejarlo correr, o moderado (2.0-3.0) para asegurar ganancia.",
            },
        },
    },
    "J": {
        "key": "J",
        "name": "ADX (Fuerza de Tendencia)",
        "icon": "\U0001f4aa",
        "category": "Seguimiento de Tendencia",
        "short_desc": "Mide que tan fuerte es la tendencia actual antes de operar.",
        "long_desc": (
            "El ADX no te dice si el precio sube o baja, sino QUE TAN FUERTE es "
            "el movimiento actual. Es como un medidor de potencia del mercado. "
            "Va de 0 a 100: por debajo de 25 no hay tendencia (el mercado esta "
            "indeciso), por encima de 25 hay tendencia, y por encima de 50 la "
            "tendencia es muy fuerte. Para saber la DIRECCION usa dos lineas "
            "auxiliares: +DI (fuerza de los compradores) y -DI (fuerza de los "
            "vendedores). Si +DI esta arriba de -DI, la tendencia es alcista. "
            "Si -DI esta arriba, es bajista. La gracia del ADX es que SOLO opera "
            "cuando la tendencia es fuerte, evitando operar en mercados laterales "
            "donde la mayoria de estrategias pierden plata."
        ),
        "when_works": "Es el filtro perfecto para evitar mercados sin direccion. Funciona excelente combinado con otras estrategias: 'solo operar si el ADX dice que hay tendencia'. Ideal en BTC/ETH cuando hay movimientos fuertes.",
        "when_fails": "Es lento para reaccionar al inicio de una tendencia (necesita que el ADX suba de 25, lo cual tarda varias velas). Tambien puede dar senales tardias: cuando el ADX esta en 50+ a veces la tendencia ya esta por terminar.",
        "example": "BTC empieza a subir. El ADX pasa de 18 a 28 y +DI (42) esta por encima de -DI (15). La estrategia confirma: hay tendencia alcista fuerte, recomienda COMPRAR. Si el ADX baja de 25, se cierra la posicion.",
        "default_params": {
            "adx_period": {
                "value": 14, "min": 7, "max": 50,
                "desc": "Cuantas velas usa para calcular el ADX.",
                "tip": "14 es el clasico de Welles Wilder. Con 7-10 reacciona mas rapido (detecta tendencias antes pero mas ruido). Con 20+ es mas suave y confiable. Para crypto volatil, 10-14 es ideal.",
            },
            "adx_threshold": {
                "value": 25, "min": 15, "max": 40,
                "desc": "Nivel minimo de ADX para considerar que hay tendencia y generar senal.",
                "tip": "25 es el clasico. Con 20 detectas tendencias mas debiles (mas senales, menos fiables). Con 30+ solo operas en tendencias claras y fuertes. Si tu par es muy volatil (ej: memecoins), subi a 30.",
            },
            "di_confirm": {
                "value": 1, "min": 1, "max": 5,
                "desc": "Cuantas velas debe mantenerse +DI por encima de -DI (o viceversa) para confirmar.",
                "tip": "Con 1 reacciona inmediatamente al cruce de DI. Con 2-3 espera confirmacion y filtra cruces falsos. Para timeframes cortos (1m-5m) usa 1. Para 1h+ podes usar 2.",
            },
            "require_rising": {
                "value": False,
                "desc": "Si se activa, solo genera senal cuando el ADX esta subiendo (la tendencia se fortalece).",
                "tip": "MUY recomendado para evitar entrar cuando la tendencia ya esta debilitandose. Un ADX de 40 pero bajando significa que la tendencia pierde fuerza, no conviene entrar.",
            },
            "sl_multiplier": {
                "value": 1.5, "min": 0.5, "max": 5.0, "step": 0.1,
                "desc": "Distancia del stop-loss como multiplo de la volatilidad (ATR).",
                "tip": "El ADX ya filtra por tendencia, asi que podes dar espacio al precio (1.5-2.5). Si el ADX es alto (40+) la tendencia es fuerte y un SL mas amplio (2.0-3.0) evita que te saquen en un retroceso normal.",
            },
            "tp_multiplier": {
                "value": 2.5, "min": 0.5, "max": 10.0, "step": 0.1,
                "desc": "Distancia del take-profit como multiplo de la volatilidad.",
                "tip": "En tendencias fuertes (ADX 40+) podes ser ambicioso (3.0-5.0). Si el ADX es justo 25-30, un TP moderado (2.0-2.5) es mas realista.",
            },
        },
    },
    "K": {
        "key": "K",
        "name": "Fear & Greed (Sentimiento)",
        "icon": "\U0001f628",
        "category": "Sentimiento de Mercado",
        "short_desc": "Usa el indice Fear & Greed como indicador contrario de sentimiento.",
        "long_desc": (
            "El Fear & Greed Index mide el sentimiento general del mercado cripto "
            "en una escala de 0 (miedo extremo) a 100 (codicia extrema). Esta "
            "estrategia es CONTRARIA: cuando todos tienen miedo, suele ser buen "
            "momento para comprar (el mercado esta sobrevendido emocionalmente). "
            "Cuando todos tienen codicia, suele ser momento de vender (el mercado "
            "esta sobrecomprado). El indice combina volatilidad, volumen, redes "
            "sociales, encuestas, dominancia de BTC y tendencias de Google. NO "
            "analiza el precio del par especifico, sino el sentimiento GENERAL del "
            "mercado cripto. Por eso funciona mejor en pares grandes (BTC, ETH) que "
            "correlacionan fuerte con el sentimiento general."
        ),
        "when_works": "Excelente en extremos de mercado: cuando el FNG esta en 10-20 (panico) historicamente es zona de acumulacion. Cuando esta en 80-90 (euforia) es zona de distribucion. Ideal como filtro complementario a estrategias tecnicas.",
        "when_fails": "En mercados laterales el FNG puede mantenerse en zona neutral (40-60) por semanas sin dar senales. Tampoco funciona bien en altcoins pequenas que no correlacionan con el sentimiento general. Y en tendencias fuertes, el mercado puede seguir subiendo con FNG en 80+ durante semanas.",
        "example": "BTC cae un 30% en una semana. El FNG baja a 15 (Miedo Extremo). La estrategia detecta que estamos en zona de panico y recomienda COMPRAR. Historicamente, el 85% de las veces que el FNG toco 15, BTC subio en las siguientes 2 semanas.",
        "default_params": {
            "fear_threshold": {
                "value": 25, "min": 5, "max": 45,
                "desc": "FNG por debajo de este valor genera senal de COMPRA.",
                "tip": "25 captura Miedo Extremo clasico. Con 10-15 solo operas en panico total (muy pocas senales pero muy confiables). Con 30-40 operas mas seguido pero con menos precision.",
            },
            "greed_threshold": {
                "value": 75, "min": 55, "max": 95,
                "desc": "FNG por encima de este valor genera senal de VENTA.",
                "tip": "75 captura Codicia Extrema. Con 80-90 solo vendes en euforia maxima. Con 60-70 vendes antes pero te podes perder subas fuertes.",
            },
            "trend_days": {
                "value": 7, "min": 3, "max": 30,
                "desc": "Cuantos dias de historial del FNG se usan para calcular la tendencia.",
                "tip": "7 dias da una buena lectura de la tendencia del sentimiento. Con 3 reacciona rapido a cambios, con 14-30 da una vision mas estable.",
            },
            "sl_multiplier": {
                "value": 1.5, "min": 0.5, "max": 5.0, "step": 0.1,
                "desc": "Distancia del stop-loss como multiplo del ATR.",
                "tip": "Como el FNG es un indicador lento (cambia una vez al dia), conviene dar espacio al precio (1.5-3.0).",
            },
            "tp_multiplier": {
                "value": 2.5, "min": 0.5, "max": 10.0, "step": 0.1,
                "desc": "Distancia del take-profit como multiplo del ATR.",
                "tip": "En extremos de sentimiento las reversiones suelen ser fuertes. Un TP de 3.0-5.0 puede capturar buenas ganancias en rebotes de panico.",
            },
        },
    },
    "L": {
        "key": "L",
        "name": "Pendiente EMA (Slope)",
        "icon": "\U0001f4d0",
        "category": "Seguimiento de Tendencia",
        "short_desc": "Mide el angulo de inclinacion de la EMA rapida para detectar fuerza y direccion de tendencia.",
        "long_desc": (
            "Imaginate que la EMA rapida es una ruta de montana. Si la ruta sube "
            "empinada, la tendencia alcista es fuerte. Si es plana, el mercado esta "
            "lateralizando. Si baja, es bajista. Esta estrategia calcula exactamente "
            "eso: el angulo de inclinacion de la EMA rapida respecto a la horizontal. "
            "Usa regresion lineal sobre los ultimos N puntos de la EMA para obtener "
            "una pendiente estable (no se distorsiona por una sola vela rara). Luego "
            "convierte esa pendiente en un angulo en grados usando arcotangente, y lo "
            "normaliza a un factor de -100 a +100 segun un angulo maximo configurable. "
            "Un factor de +80 significa 'la EMA sube con mucha fuerza'. Un factor de "
            "-20 significa 'la EMA baja suavemente'. Cero = la EMA esta plana. "
            "Trabaja sobre la EMA (no el precio crudo) porque la EMA ya filtra el "
            "ruido intravela, asi que la pendiente refleja la tendencia real."
        ),
        "when_works": "Ideal como filtro de fuerza (rol 'Fuerza') combinado con otras estrategias direccionales. Confirma que la tendencia tiene momentum real antes de operar. Funciona muy bien en timeframes de 5m a 4h.",
        "when_fails": "En mercados con alta volatilidad y sin tendencia definida, la pendiente oscila rapidamente. Tambien puede dar senales tardias al inicio de un movimiento porque la EMA tarda en reaccionar.",
        "example": "BTC sube sostenidamente de $95.000 a $97.000 en 4 horas. La EMA(9) en velas de 15m muestra un angulo de +18 grados. Con max_angle=45, el factor es +40. La estrategia recomienda COMPRAR con confianza 40%. Si ademas la usas como fuerza, su force=0.40 confirma tendencia moderada.",
        "default_params": {
            "ema_period": {
                "value": 9, "min": 3, "max": 50,
                "desc": "Periodo de la EMA rapida sobre la que se calcula la pendiente.",
                "tip": "EMA corta (5-9) reacciona rapido, ideal para scalping/intraday. EMA larga (20-50) da tendencias mas suaves, mejor para swing. El default de 9 es un buen balance.",
            },
            "slope_window": {
                "value": 10, "min": 3, "max": 30,
                "desc": "Cuantos puntos de la EMA usa para calcular la pendiente via regresion lineal.",
                "tip": "Con 5 la pendiente reacciona rapido pero es mas ruidosa. Con 20 es muy estable pero lenta. El default de 10 da un angulo suave sin demasiado retraso.",
            },
            "max_angle": {
                "value": 45.0, "min": 15.0, "max": 90.0, "step": 5.0,
                "desc": "Angulo maximo esperado en grados. Controla la sensibilidad de la normalizacion.",
                "tip": "Con 30 grados, una pendiente moderada ya da factor alto (mas sensible, ideal para scalping en 1m-5m). Con 60 grados necesitas pendientes fuertes para llegar a factor 100 (menos sensible, mejor para swing en 1h-4h). 45 es un buen default general.",
            },
            "smooth_period": {
                "value": 5, "min": 1, "max": 20,
                "desc": "Cuantas evaluaciones pasadas se promedian para suavizar el angulo y evitar picos aislados.",
                "tip": "Con 1 no hay suavizado (reacciona instantaneamente). Con 5 promedia las ultimas 5 mediciones (filtra picos). Con 10+ es muy suave pero lento. Recomendado: 3-7.",
            },
            "deadzone": {
                "value": 5.0, "min": 0.0, "max": 30.0, "step": 1.0,
                "desc": "Factor minimo (en valor absoluto) para generar senal. Por debajo = ESPERAR.",
                "tip": "Con 0 siempre genera senal (incluso con la EMA apenas inclinada). Con 5-10 filtra los momentos de lateralizacion. Con 20+ solo reacciona a pendientes muy claras.",
            },
            "sl_multiplier": {
                "value": 1.5, "min": 0.5, "max": 5.0, "step": 0.1,
                "desc": "Distancia del stop-loss como multiplo de la volatilidad (ATR).",
                "tip": "La pendiente de EMA es una senal de tendencia. Dale espacio al precio (1.5-2.5) para que la tendencia se desarrolle.",
            },
            "tp_multiplier": {
                "value": 2.5, "min": 0.5, "max": 10.0, "step": 0.1,
                "desc": "Distancia del take-profit como multiplo de la volatilidad.",
                "tip": "Con pendiente fuerte (factor alto), podes ser ambicioso (3.0-5.0). Con pendiente suave, un TP moderado (2.0-2.5) es mas realista.",
            },
        },
    },
}
