"""Offline plots of existing evidence. Never loads a checkpoint or runs inference."""
import argparse
from collections import defaultdict
import hashlib
import html
import json
import math
import os
from pathlib import Path
import statistics
import textwrap

from asi import ROOT

MIB = 1024**2
LABELS = {
    'native': 'Modelo completo', 'resident': 'Modelo completo',
    'lru': 'Carga cuando hace falta', 'popularity': 'Precarga los más usados',
    'semantic': 'Precarga según el tema', 'learned': 'Predictor de uso',
    'prefetch': 'Caché exacta', 'restrict': 'Candidatos limitados',
    'oracle_uniform': 'Tema conocido · 2 fijos',
    'predicted_uniform': 'Tema predicho · 2 fijos',
    'global_uniform': 'Siempre los mismos 2',
    'oracle_calibrated': 'Tema conocido · mezcla ajustada',
    'restricted': 'Solo pools elegidos', 'unrestricted': 'Todos los pools',
    'ablated': 'Excluir pools elegidos',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def memory_metrics(memory):
    """Snapshots, not continuous peaks. RAM aliases are counted only once."""
    m = memory.get('inventory', memory)
    experts = m.get('experts')
    gpu = [e for e in experts if e['device'].startswith('cuda')] if experts is not None else None
    backing = m.get('ram_backing_bytes')
    if backing is None and experts is not None and all('ram_backing_bytes' in e for e in experts):
        backing = sum(e['ram_backing_bytes'] for e in experts)
    allocated = m.get('cuda', {}).get('allocated_bytes')
    return {'gpu_experts': len(gpu) if gpu is not None else None,
            'gpu_expert_mib': sum(e.get('weight_bytes', e.get('execution_weight_bytes', 0)) for e in gpu)/MIB if gpu is not None else None,
            'ram_expert_mib': backing/MIB if backing is not None else None,
            'cuda_allocated_mib': allocated/MIB if allocated is not None else None}


def route_changes(rows, phase='decode'):
    """Count layer-pair changes, ignoring pair ordering. Reset between turns."""
    changes, comparisons = 0, 0
    for row in rows:
        previous = None
        for event in row['events']:
            current = {str(layer): frozenset(ids[0]) for layer, ids in event['routes'].items()}
            if previous is not None and event['phase'] == phase:
                for layer in current.keys() & previous.keys():
                    comparisons += 1
                    changes += current[layer] != previous[layer]
            previous = current
    return changes, comparisons


def event_metrics(rows, phase='decode'):
    events = [e for r in rows for e in r['events'] if e['phase'] == phase]
    n = len(events)
    seconds = sum(e['seconds'] for e in events)
    loads = sum(e['cache_delta']['misses'] for e in events)
    changes, comparisons = route_changes(rows, phase)
    return {'steps': n, 'loads': loads,
            'evictions': sum(e['cache_delta']['evictions'] for e in events),
            'h2d_mib_per_step': sum(e['cache_delta']['host_to_device_bytes'] for e in events)/MIB/n if n else None,
            'no_load_percent': 100*sum(e['cache_delta']['misses'] == 0 for e in events)/n if n else None,
            'ms_median': 1000*median([e['seconds'] for e in events]) if n else None,
            'steps_per_second': n/seconds if seconds else None,
            'pair_changes': changes, 'pair_comparisons': comparisons,
            'prepare_mib': sum(r['prefetch_delta']['host_to_device_bytes'] for r in rows)/MIB}


class Gallery:
    def __init__(self, output):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'axes.titleweight': 'bold', 'figure.facecolor': '#f4f7fb',
                             'axes.facecolor': '#ffffff', 'savefig.facecolor': '#f4f7fb'})
        self.plt = plt
        self.output = output
        self.cards, self.sources, self.data = [], [], []

    def source(self, path, status, note=''):
        self.sources.append({'path': str(path.resolve()), 'sha256': sha(path), 'status': status, 'note': note})

    def save(self, fig, title, note, source, data=None):
        name = f'{len(self.cards)+1:02d}'
        fig.savefig(self.output/f'{name}.png', dpi=155)
        fig.savefig(self.output/f'{name}.svg')
        self.plt.close(fig)
        self.cards.append({'id': name, 'title': title, 'note': note, 'source': str(source), 'png': name+'.png'})
        self.data.append({'figure': name, 'source': str(source), 'data': data})

    def bars(self, title, modes, panels, note, source):
        ncols = 2
        nrows = math.ceil(len(panels)/ncols)
        fig, axes = self.plt.subplots(nrows, ncols, figsize=(15, 1.2+nrows*3.3), squeeze=False)
        fig.suptitle(title, fontsize=17, x=.035, ha='left', y=.985)
        names = ['\n'.join(textwrap.wrap(LABELS.get(m, m), 27)) for m in modes]
        colors = ['#247b92' if m in ('native','resident','unrestricted') else '#dc9146' if 'fixed' in m or 'uniform' in m else '#5666a5' for m in modes]
        for ax, (heading, unit, values) in zip(axes.flat, panels):
            ax.set_title(heading, loc='left', fontsize=11, pad=12)
            nums = [v if v is not None else 0 for v in values]
            ax.barh(range(len(modes)), nums, color=colors, height=.6)
            ax.set_yticks(range(len(modes)), names)
            ax.invert_yaxis(); ax.set_xlabel(unit)
            high = max(nums, default=0)
            ax.set_xlim(0, max(high*1.22, 1))
            for i, value in enumerate(values):
                label = 'No medido' if value is None else f'{value:,.2f}'.replace(',', ' ')
                ax.text((value or 0)+max(high*.018, .02), i, label, va='center', fontsize=9)
            ax.set_axisbelow(True); ax.grid(axis='x', alpha=.15)
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        fig.subplots_adjust(left=.18, right=.98, top=.89, bottom=.1, hspace=.7, wspace=.85)
        fig.text(.035, .015, 'Alcance, unidades y fuentes en el panel HTML. Acierto de tokens no significa respuestas correctas.', fontsize=9, color='#4a5364')
        self.save(fig, title, note, source, {'modes': modes, 'panels': panels})

    def memory(self, title, samples, source):
        modes = list(samples)
        extracted = {mode: [memory_metrics(m) for m in memories] for mode, memories in samples.items()}
        panels = [(label, unit, [median([r[key] for r in extracted[m]]) for m in modes]) for key, label, unit in (
            ('gpu_experts','Expertos en GPU al registrar la memoria','Instancias de expertos, sumando capas'),
            ('gpu_expert_mib','Pesos de expertos en GPU','MiB; no es la VRAM total'),
            ('ram_expert_mib','Copia de respaldo de expertos en RAM','MiB; no es la RAM total del proceso'),
            ('cuda_allocated_mib','Memoria asignada por PyTorch en GPU','MiB; instantánea, no pico'))]
        self.bars(title, modes, panels, 'Mediana de las instantáneas disponibles. No medido significa que el informe no guardó ese dato, nunca cero. '
                  'Backbone, expertos compartidos, KV y temporales también ocupan memoria. RAM puede conservar copia de expertos que están en GPU.', source)


def draw_report(g, path):
    d = read(path); summary = d.get('summary', {})
    study = path.parent.name
    if 'aggregate' in d and 'results' in d:
        modes = list(d['aggregate'])
        rows = {m: [r for r in d['results'] if r['mode'] == m] for m in modes}
        g.bars('Auditoría de pools · '+study, modes, [
            ('Error de predicción; menor es mejor', 'NLL; muestra de diagnóstico', [d['aggregate'][m]['token_weighted_nll'] for m in modes]),
            ('Violaciones del permiso de expertos', 'Total registrado', [sum(r['mask_violations'] for r in rows[m]) for m in modes]),
            ('Tiempo de cálculo, sin trazado', 'Milisegundos por entrada; mediana', [1000*median([r['forward_seconds_without_trace'] for r in rows[m]]) for m in modes]),
            ('Expertos expulsados de GPU', 'Total de evicciones', [sum(r['cache_delta'].get('evictions',0) for r in rows[m]) for m in modes])],
            f"Checkpoint paso {d.get('metadata',{}).get('step', 'aleatorio')}. Prueba de funcionamiento, no de competencia. "
            'Excluir pools es una intervención deliberada. Cero violaciones solo confirma que se respetaron los permisos.', path)
        g.memory('Memoria de la auditoría · '+study, {m:[r['memory'] for r in rows[m]] for m in modes}, path)
    elif 'n_experts_per_class_per_layer' in d:
        modes = list(summary)
        g.bars('Dos expertos fijos · calidad de predicción', modes, [
            ('Acierto del siguiente fragmento de texto', '% de tokens acertados; mayor es mejor', [summary[m]['token_accuracy']*100 for m in modes]),
            ('Error de predicción', 'NLL; menor es mejor', [summary[m]['nll'] for m in modes]),
            ('Dificultad respecto al modelo completo', 'Multiplicador de perplexity; 1 = control', [summary[m]['perplexity_ratio_to_native'] for m in modes]),
            ('Tiempo de cálculo de una ventana', 'Milisegundos; mediana, sin clasificación/precarga', [summary[m]['forward_p50_ms'] for m in modes])],
            f"{summary[modes[0]]['windows']} ventanas, {summary[modes[0]]['tokens']} tokens por variante; mismas entradas. No son porcentajes de respuestas correctas. "
            'Tiempo de un piloto, no una comparación repetida de velocidad. Tema conocido = oracle; tema predicho = clasificador. '
            'Mezcla ajustada conserva una escala media aprendida en train, pero sigue sin calcular E.', path)
        g.memory('Dos expertos fijos · memoria observada', {m:[v] for m,v in d['memory'].items()}, path)
        modes = list(summary)
        g.bars('Dos expertos fijos · movimiento al cambiar de clase', modes, [
            ('Antes de procesar cada ventana', 'MiB transferidos en total en este ensayo', [summary[m]['prefetch_h2d_bytes']/MIB for m in modes]),
            ('Durante el cálculo de las ventanas', 'MiB transferidos en total', [summary[m]['forward_h2d_bytes']/MIB for m in modes])],
            'Preparar una clase puede cargar pesos; mantenerlos fijos evita cargas durante esa ventana. El total depende del orden de las clases. '
            'La carga inicial del modelo completo no está incluida; su cero no significa que nunca se hayan copiado pesos a GPU.', path)
        draw_generation(g, path.parent/'generation.jsonl', fixed=True)
    elif summary and 'decode' in next(iter(summary.values())):
        draw_generation(g, path.parent/'turns.jsonl', fixed=False)
        samples = defaultdict(list)
        for key, value in d['memory'].items(): samples[key.split(':')[1]].append(value)
        g.memory('Dos plazas con E · memoria observada', samples, path)
    elif summary and 'pipeline_p50_seconds' in next(iter(summary.values())):
        modes = list(summary)
        g.bars('Precarga de expertos · caché de 52 plazas', modes, [
            ('Transferencias RAM → GPU', 'MiB totales; incluye precargas', [summary[m]['cache_totals']['host_to_device_bytes']/MIB for m in modes]),
            ('Tiempo por turno, incluyendo preparación', 'Milisegundos; mediana', [summary[m]['pipeline_p50_seconds']*1000 for m in modes]),
            ('Cargas al necesitar un experto ausente', 'Total de misses; no incluye precargas', [summary[m]['cache_totals']['misses'] for m in modes]),
            ('Diferencias frente a la ejecución completa', 'Turnos con argmax diferente', [summary[m]['argmax_mismatch_turns'] for m in modes])],
            f"{summary[modes[0]]['turn_observations']} observaciones por política; incluyen repeticiones de las mismas conversaciones. Se procesan prompts; aquí no se generan respuestas. "
            'Mismas predicciones no significa respuestas verdaderas. El predictor no superó a la popularidad global en este piloto.', path)
        samples = defaultdict(list)
        for key, value in d['memory'].items(): samples[key.split(':')[1]].append(value)
        g.memory('Caché de 52 plazas · memoria observada', samples, path)
    elif summary and 'baseline_nll' in next(iter(summary.values())):
        modes = ['resident']+list(summary)
        first = next(iter(summary.values()))
        g.bars('Primera clasificación de expertos · cuatro candidatos por clase', modes, [
            ('Error de predicción', 'NLL; menor es mejor', [first['baseline_nll']]+[v['nll'] for v in summary.values()]),
            ('Tiempo de cálculo por ventana', 'Milisegundos; media, sin precarga', [first['mean_baseline_forward_seconds']*1000]+[v['mean_forward_seconds']*1000 for v in summary.values()]),
            ('Selecciones distintas de las originales', 'Slots de routing diferentes; no errores de software', [0]+[v['route_mismatches'] for v in summary.values()]),
            ('Cambios físicos en la caché', 'Evicciones en las ventanas registradas', [None]+[sum(r['cache_delta'].get('evictions',0) for r in d['results'] if r['policy']==m) for m in summary])],
            f"{first['windows']} ventanas; consultar cobertura por dominio en el informe. Restricción significa permitir cuatro candidatos, no tener exactamente cuatro expertos "
            'por capa en GPU. La caché tiene un límite global de 52. No comparar esta NLL con la de otra muestra.', path)
        g.memory('Primer ensayo · memoria observada', {m:[v] for m,v in d['memory'].items()}, path)
    else:
        return False
    return True


def draw_generation(g, path, fixed):
    if not path.exists():
        return
    rows = jsonl(path); field = 'mode' if fixed else 'policy'
    modes = list(dict.fromkeys(r[field] for r in rows))
    groups = {m:[r for r in rows if r[field] == m] for m in modes}
    values = {m:event_metrics(groups[m]) for m in modes}
    g.source(path, 'trazas de generación')
    title = 'Generación con dos expertos fijos, sin E' if fixed else 'Generación con dos plazas por capa, conservando E'
    g.bars(title, modes, [
        ('Pasos de generación sin cargar expertos', '%; 100 significa que no hubo cargas durante decode', [values[m]['no_load_percent'] for m in modes]),
        ('Datos transferidos durante generación', 'MiB por paso de decode', [values[m]['h2d_mib_per_step'] for m in modes]),
        ('Tiempo de cálculo de un paso', 'Milisegundos; mediana', [values[m]['ms_median'] for m in modes]),
        ('Velocidad del cálculo de decode', 'Pasos / suma de segundos; excluye prefill y preparación', [values[m]['steps_per_second'] for m in modes])],
        (f'{len(groups[modes[0]])} turnos por variante; historias generadas pueden divergir.' if fixed else
         f"{len(groups[modes[0]])} turnos por variante, con repeticiones. Turnos con tokens distintos del control: {sum(not r['output_equal'] for r in rows)}; con rutas distintas: {sum(not r['routes_equal'] for r in rows)}.")+
        ' El primer token sale del prefill: generar 16 tokens requiere solo 15 pasos de decode. Comparar tiempos solo dentro del ensayo.', path)
    g.bars(title+' · cambios y cargas', modes, [
        ('Veces que cambia la pareja elegida', 'Suma por capa entre pasos de decode', [values[m]['pair_changes'] for m in modes]),
        ('Cargas de expertos durante decode', 'Total de misses', [values[m]['loads'] for m in modes]),
        ('Expulsiones de GPU durante decode', 'Total de evicciones', [values[m]['evictions'] for m in modes]),
        ('Datos cargados al preparar los turnos', 'MiB totales antes del prefill', [values[m]['prepare_mib'] for m in modes])],
        'Elegir otra pareja no siempre mueve pesos: si todos residen en GPU, cambia la selección pero no hay swapping. '
        'Una carga inicial tampoco implica expulsión. El cambio de pareja ignora el orden de los dos expertos y no compara turnos distintos.', path)
    # One explicitly identified example, not an average or cherry-picked optimum.
    chosen_mode = 'predicted_uniform' if fixed else 'lru'
    if chosen_mode not in groups:
        return
    row = groups[chosen_mode][0]; events = row['events']
    layers = sorted(events[0]['routes'], key=int)
    fig, axes = g.plt.subplots(2,1,figsize=(14,6.6), gridspec_kw={'height_ratios':[2,1]})
    # Color encodes pair identity uniquely: a*E+b for sorted a,b, not residency.
    pairs = [[tuple(sorted(event['routes'][layer][0])) for event in events] for layer in layers]
    unique = sorted({pair for layer in pairs for pair in layer})
    ids = {pair:i for i,pair in enumerate(unique)}
    axes[0].imshow([[ids[pair] for pair in layer] for layer in pairs], aspect='auto', interpolation='nearest', cmap='turbo', vmin=0, vmax=max(len(ids)-1,1))
    axes[0].set_yticks(range(len(layers)), layers); axes[0].set_ylabel('Capa con expertos')
    axes[0].set_title('Color = identidad de la pareja (dentro de esta figura); una fila uniforme mantiene sus expertos',fontsize=11)
    loads = [e['cache_delta']['host_to_device_bytes']/MIB for e in events]
    axes[1].bar(range(len(events)), loads,color='#dc9146'); axes[1].set_ylabel('RAM → GPU (MiB)')
    axes[1].set_xlabel('Paso de cálculo: primero lectura de la pregunta, después generación')
    boundary = next((i for i,e in enumerate(events) if e['phase']=='decode'),None)
    if boundary is not None:
        for ax in axes: ax.axvline(boundary-.5, color='#1d2636', linestyle='--', linewidth=1)
    caption = f"{LABELS[chosen_mode]} · primera traza: {row['session']}, turno {row['turn']}"
    fig.suptitle(caption,fontsize=15); fig.tight_layout(rect=(0,0,1,.94))
    g.save(fig, caption, 'Ejemplo individual, no resumen estadístico. La línea separa prefill y decode. '
           'La precarga previa a la primera columna no se incluye en las barras. Las parejas completas están en datos_graficas.json.', path,
           {'layer_pairs': dict(zip(layers,pairs)), 'h2d_mib':loads,'decode_boundary':boundary})


def draw_chat(g, path):
    d = read(path)
    if not isinstance(d,dict) or not d.get('turns') or not all('cache_delta' in r and 'memory' in r for r in d['turns']):
        return False
    rows = d['turns']; names = [f'Turno {i+1}' for i in range(len(rows))]
    g.bars('Prueba de chat · '+path.parent.name+' / '+path.stem, names, [
        ('Transferencias RAM → GPU', 'MiB por turno; incluye preparación', [r['cache_delta']['host_to_device_bytes']/MIB for r in rows]),
        ('Tiempo de generación registrado', 'Segundos; no compara motores de generación diferentes', [r.get('generation_seconds') for r in rows]),
        ('Expertos en GPU al terminar el turno', 'Instancias de expertos, sumando capas', [memory_metrics(r['memory'])['gpu_experts'] for r in rows]),
        ('RAM de respaldo de expertos', 'MiB; no es RSS ni caché del sistema operativo', [r['cache'].get('ram_backing_bytes',0)/MIB if 'ram_backing_bytes' in r['cache'] else None for r in rows])],
        'Prueba de integración, no benchmark de corrección. Cada figura mantiene separados sus turnos. '
        'Con almacenamiento en disco, los bytes leídos son lecturas lógicas; no demuestran lecturas físicas del SSD.', path)
    return True


def write_html(g):
    e = html.escape
    def link(path):
        return e(Path(os.path.relpath(path,g.output)).as_posix(), quote=True)
    glossary = [('original','El modelo tal como se entrenó, antes de cambiar cómo elegimos expertos.'),
                ('classified','El modelo ya entrenado cuyos expertos hemos etiquetado después observando qué temas los activan. No es otro entrenamiento por sí mismo.'),
                ('domain','El modelo entrenado desde el principio con grupos de expertos asignados a temas. Domain significa tema o área.'),
                ('oracle','Le damos la etiqueta conocida: «este texto es de matemáticas». Sirve para separar el error al reconocer el tema del error del modelo. No sabe la respuesta correcta.'),
                ('predicted','Un clasificador intenta adivinar el tema de la pregunta.'),
                ('E / router','El mecanismo interno que puntúa expertos para cada token. Es distinto del clasificador externo de temas.'),
                ('token','Un fragmento de texto; no necesariamente una palabra.'),
                ('prefill / decode','Leer la pregunta / ir calculando los siguientes fragmentos de la respuesta.'),
                ('hit / miss','El experto solicitado ya está en GPU / hay que cargarlo.'),
                ('swapping','Mover pesos entre RAM y GPU para sustituir expertos.'),
                ('evicción','Sacar un experto de GPU para dejar sitio. No siempre coincide con una carga.'),
                ('NLL / perplexity','Medidas del error al predecir texto. Menor es mejor sobre la misma prueba. No equivalen a respuestas verdaderas.')]
    rows=''.join(f'<tr><th>{e(a)}</th><td>{e(b)}</td></tr>' for a,b in glossary)
    cards=''.join(f'<section id="fig-{c["id"]}"><h2>{e(c["title"])}</h2><p>{e(c["note"])}</p>'
                  f'<a href="{c["png"]}"><img loading="lazy" src="{c["png"]}" alt="{e(c["title"])}"></a>'
                  f'<p><a href="{c["id"]}.svg">SVG para el TFG</a> · <a href="{link(c["source"])}">Datos originales</a></p></section>' for c in g.cards)
    sources=''.join(f'<tr><td><a href="{link(s["path"])}">{e(str(Path(s["path"]).relative_to(ROOT)) if Path(s["path"]).is_relative_to(ROOT) else s["path"])}</a></td><td>{e(s["status"])}</td><td>{e(s["note"])}</td></tr>' for s in g.sources)
    nav=''.join(f'<li><a href="#fig-{c["id"]}">{e(c["title"])}</a></li>' for c in g.cards)
    content=f'''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASI · Entender el proyecto y sus pruebas</title><style>
body{{margin:0;background:#edf2f8;color:#202c40;font:17px/1.65 system-ui,sans-serif}}main{{max-width:1250px;margin:auto;padding:32px}}
h1{{font-size:36px;line-height:1.2}}h2{{font-size:25px}}section,.intro{{background:white;border-radius:14px;padding:26px;margin:24px 0;box-shadow:0 2px 8px #0001}}
a{{color:#185b89}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dbe3ee;overflow-wrap:anywhere}}
.notice{{border-left:5px solid #dc9146;padding:10px 20px;background:#fff7e9}}details{{margin:15px 0}}li{{margin:6px 0}}
@media(max-width:700px){{main{{padding:12px}}section,.intro{{padding:15px}}h1{{font-size:28px}}}}
</style><main><h1>Un modelo grande, con pocos expertos en GPU</h1>
<p>Este panel vuelve a dibujar datos ya guardados. No entrena, no abre checkpoints y no ejecuta evaluaciones nuevas.</p>
<div class="intro"><h2>Lo que estamos investigando</h2><p>Queremos que un modelo pueda trabajar con menos memoria de GPU.
La idea es dejar muchos expertos en RAM y llevar a GPU solo los necesarios para la conversación.</p>
<p><strong>Elegir dos expertos no significa que solo dos ocupen memoria.</strong> El modelo normal usa dos por capa y token,
pero puede mantener los ocho de cada capa cargados. Nuestro límite estricto deja dos cargados en cada una de las once capas con expertos: 22 instancias en total.</p>
<p>Hasta ahora, conservar las decisiones originales mantiene sus predicciones pero sigue moviendo expertos durante la respuesta.
Fijar una pareja por tema elimina esas cargas internas, aunque pierde calidad de predicción.</p>
<div class="notice">La comparación extensa de tres modelos está preparada, <strong>no ejecutada</strong>. No aparece como resultado.
Los ensayos usan muestras y motores diferentes: comparar velocidad o NLL entre figuras puede llevar a una conclusión falsa.</div>
<p><a href="{link(ROOT/'docs/RECAPITULACION.md')}">Leer la recapitulación completa</a> · <a href="datos_graficas.json">Valores de las gráficas</a> · <a href="fuentes.json">Inventario y hashes</a></p>
<details open><summary><strong>Traducción de los nombres</strong></summary><table>{rows}</table></details>
<details><summary><strong>Ir a una gráfica ({len(g.cards)})</strong></summary><ol>{nav}</ol></details></div>
{cards}<section><h2>Qué datos se encontraron</h2><p>Los backups duplicados no se cuentan como pruebas nuevas. Los esquemas sin adaptador se señalan, no se inventan métricas.</p><table><tr><th>Archivo</th><th>Estado</th><th>Alcance</th></tr>{sources}</table></section></main></html>'''
    (g.output/'index.html').write_text(content,encoding='utf-8')


def draw_overview(g, results):
    paths = [results/name/'report.json' for name in ('posthoc_04750','decode_two_experts_04750_v1','fixed_two_no_E_04750_v1')]
    if not all(p.exists() for p in paths): return
    reports = [read(p) for p in paths]
    hashes = {d['metadata'].get('checkpoint_sha256') for d in reports}
    if len(hashes) != 1 or None in hashes: return
    posthoc, decode, fixed = reports
    first = posthoc['memory']['prefetch']['inventory']
    total = sum(e['weight_bytes'] for e in first['experts'])/MIB
    samples = [memory_metrics(first),
               memory_metrics(next(v for k,v in decode['memory'].items() if ':lru:' in k)),
               memory_metrics(fixed['memory']['predicted_uniform'])]
    modes = ['Todos los expertos','Caché de 52 plazas','Dos por capa, con E','Dos por capa, fijos']
    g.bars('La diferencia entre usar expertos y tenerlos cargados', modes, [
        ('Expertos cargados en GPU', 'Instancias sumadas en las 11 capas con expertos', [len(first['experts'])]+[v['gpu_experts'] for v in samples]),
        ('Solo sus pesos en GPU', 'MiB; las partes comunes y el contexto se añaden aparte', [total]+[v['gpu_expert_mib'] for v in samples])],
        'Comparación del mismo checkpoint (hash verificado). Se muestran instantáneas de distintas pruebas, válidas para comparar estos pesos, '
        'no para comparar tiempos o calidad. El control completo carga todas las instancias del inventario. '
        'Las dos últimas opciones ocupan lo mismo, pero con E las parejas cambian entre tokens; con pareja fija se mantienen durante la respuesta.', paths[0])


def build(results, output, refresh=False):
    results, output = Path(results).resolve(), Path(output).resolve()
    if not results.is_dir():
        raise FileNotFoundError(results)
    if refresh and output.exists() and not all((output/name).is_file() for name in ('index.html','fuentes.json')):
        raise ValueError('Refresh requires a previously generated gallery; use a new output directory')
    output.mkdir(parents=True,exist_ok=refresh)
    g = Gallery(output); seen = set()
    draw_overview(g, results)
    # Originals first, then unique regression results in backups.
    priority = {'fixed_two_no_E_04750_v1':0, 'decode_two_experts_04750_v1':1, 'cache_study_04750_v1':2, 'posthoc_04750':3}
    paths = sorted(results.rglob('report.json'),key=lambda p: ('refactor_backup' in p.parts,priority.get(p.parent.name,4),str(p)))
    for path in paths:
        fingerprint=sha(path)
        if fingerprint in seen:
            g.source(path,'duplicado','Copia idéntica; no se vuelve a contar.')
            continue
        seen.add(fingerprint)
        supported=draw_report(g,path)
        g.source(path,'graficado' if supported else 'sin adaptador','Resultados conservados; sin mezclar protocolos.')
    for path in sorted(results.rglob('*.json')):
        if path.name.startswith('chat') or path.name in ('prompt_sessions.json','checkpoint_chat.json'):
            fingerprint=sha(path)
            if fingerprint in seen:
                g.source(path,'duplicado'); continue
            seen.add(fingerprint)
            if draw_chat(g,path): g.source(path,'chat de integración')
    for path in sorted(results.rglob('manifest.json')):
        d=read(path)
        if isinstance(d,dict) and d.get('state') in ('prepared_not_executed','models_registered_not_executed'):
            g.source(path,'pendiente; sin resultados','Entradas y matriz congeladas, no una evaluación ejecutada.')
    for path in sorted(results.glob('*/log.txt')):
        rows=jsonl(path)
        training=[r for r in rows if 'loss' in r]
        validation=[r for r in rows if 'val_loss' in r]
        if not training and not validation: continue
        fig,axes=g.plt.subplots(1,2,figsize=(13,4.5))
        for ax,field,title in zip(axes,['loss','tokens_sec'],['Error durante entrenamiento','Velocidad de entrenamiento (tokens/s)']):
            points=[r for r in training if field in r]
            ax.plot([r['step'] for r in points],[r[field] for r in points],marker='o',color='#247b92',label='train')
            if field=='loss' and validation:
                ax.plot([r['step'] for r in validation],[r['val_loss'] for r in validation],marker='s',color='#dc9146',label='validación');ax.legend()
            ax.set_title(title);ax.set_xlabel('Paso registrado');ax.grid(alpha=.2)
        fig.suptitle('Log local disponible · '+path.parent.name);fig.tight_layout(rect=(0,0,1,.92))
        g.save(fig,'Entrenamiento · solo los pasos guardados','El directorio también contiene otros checkpoints. Este log no se atribuye automáticamente al checkpoint 4750. No se reconstruyen los miles de pasos que no figuran aquí.',path,rows)
        g.source(path,'log parcial','Solo se dibujan los pasos realmente registrados.')
    (output/'datos_graficas.json').write_text(json.dumps(g.data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (output/'fuentes.json').write_text(json.dumps(g.sources,indent=2,ensure_ascii=False),encoding='utf-8')
    write_html(g)
    return {'figures':len(g.cards),'sources':len(g.sources),'index':str(output/'index.html')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--output',type=Path,default=ROOT/'results/project_overview_v1')
    parser.add_argument('--refresh',action='store_true',help='Regenerate an existing gallery; original experiment files are not edited')
    args=parser.parse_args()
    print(json.dumps(build(args.results,args.output,args.refresh),indent=2))


if __name__=='__main__': main()
