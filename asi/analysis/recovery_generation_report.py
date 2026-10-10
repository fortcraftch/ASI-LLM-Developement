"""Render matched generation outputs and resource measurements, without a GPU."""
import argparse
import html
import json
from pathlib import Path


def read_reports(directory):
    reports = [json.loads((directory/v/'report.json').read_text(encoding='utf8')) for v in ('original','32','16')]
    base = reports[0]
    for report in reports:
        if not report['completed'] or len(report['records']) != len(base['records']):
            raise ValueError('All three runs must be complete')
        for key in ('source_identity', 'corpus_identity', 'protocol', 'gpu', 'torch'):
            if report[key] != base[key]:
                raise ValueError('Incomparable '+key)
        for a, b in zip(base['records'], report['records']):
            if any(a[k] != b[k] for k in ('id','prompt','input_ids')):
                raise ValueError('Prompt identity/order differs')
    return reports


def render(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    reports = read_reports(directory)
    names = ['Original · routing libre', 'Recuperado · 32/capa', 'Recuperado · 16/capa']
    domains = ['Programación', 'Matemáticas', 'Biología']
    colors = ['#64748b','#2563eb','#db2777']
    fig, axes = plt.subplots(2, 3, figsize=(17,9))
    specs = [
        ('Tiempo total de respuesta (s)', lambda r: r['prepare_seconds']+r['wall_generation_seconds']),
        ('Velocidad de decodificación (tokens/s)', lambda r: r['decode_tokens_per_second'] or 0),
        ('Pico VRAM asignada por PyTorch (GiB)', lambda r: r['peak_cuda_allocated_bytes']/2**30),
        ('Pico RAM del proceso muestreado (GiB)', lambda r: r['sampled_peak_rss_bytes']/2**30),
        ('Transferencia RAM → GPU durante generación (GiB)', lambda r: r['generation_transfers']['host_to_device_bytes']/2**30),
        ('Trigramas de tokens repetidos (%)', lambda r: 100*r['metrics']['repeated_trigram_fraction'])]
    for ax, (title, metric) in zip(axes.flat,specs):
        for i,(report,name,color) in enumerate(zip(reports,names,colors)):
            values = [metric(r) for r in report['records']]
            bars = ax.bar(np.arange(3)+(i-1)*.25, values, width=.24, label=name, color=color)
            ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(3),domains)
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
        ax.margins(y=.22)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3)
    fig.suptitle('OLMoE: original frente a recuperación con expertos residentes',fontsize=17,y=.99)
    fig.text(.5,.015,'Greedy · máximo 256 tokens · parada EOS · 3 preguntas · una pasada · mismo equipo\n'
             'Original con caché LRU y transferencias; categorías conocidas. La longitud y la corrección de las respuestas pueden diferir.',
             ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.07,1,.90])
    fig.savefig(directory/'comparison.png',dpi=160)
    plt.close(fig)
    lines = ['# Generación: original, recuperación 32 y recuperación 16', '',
        'Tres preguntas reservadas, greedy, hasta 256 tokens y parada EOS. Todos activan 8 expertos por token y capa. '
        '32 y 16 indican expertos candidatos residentes por capa, no expertos activados simultáneamente. '
        'El original conserva su routing libre entre 64 expertos por capa y usa una caché global de 512 parejas (capa, experto); '
        'no cabe completo en esta GPU. Su tiempo incluye las transferencias necesarias en este equipo.', '',
        'Los ajustes se seleccionaron por validación: router + LoRA de expertos, paso 24; '
        '32 usa router-expert-lora y 16 usa curriculum_direct16, el mejor para la restricción 16 en validación. '
        'Cada proceso tiene un límite de caché de expertos en RAM de 2 GiB y presupuesto de proceso de 12 GiB. '
        'Las categorías se proporcionan desde el dataset (oracle); no se mide el clasificador.', '',
        '![Comparación](comparison.png)', '',
        '| Variante | Dominio | Tokens | Parada | Preparar s | Generar s | TTFT s | tokens/s decode | VRAM GiB | RAM GiB | Cargas al generar |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for report,name in zip(reports,names):
        for domain,r in zip(domains,report['records']):
            lines.append(f"| {name} | {domain} | {len(r['token_ids'])} | {r['stop_reason']} | {r['prepare_seconds']:.2f} | {r['wall_generation_seconds']:.2f} | {r['ttft_seconds']:.2f} | {(r['decode_tokens_per_second'] or 0):.2f} | {r['peak_cuda_allocated_bytes']/2**30:.2f} | {r['sampled_peak_rss_bytes']/2**30:.2f} | {r['generation_transfers']['loads']} |")
    lines += ['', '## Espacio en disco', '']
    for report,name in zip(reports,names):
        lines += [f"- {name}: modelo base {report['model_disk_bytes']/2**30:.3f} GiB; adaptación adicional {report['adapter_disk_bytes']/2**20:.1f} MiB."]
    lines += ['', 'Se comparte el mismo modelo base en disco; las restricciones no reducen su archivo original. '
              'La VRAM es el pico asignado por PyTorch en preparación + generación; el JSON conserva también memoria reservada. '
              'La RAM es RSS muestreada cada 50 ms, no solo pesos ni caché del sistema operativo. '
              'Las lecturas de disco del JSON son lógicas: pueden resolverse desde la caché del sistema.', '',
              '## Textos completos', '']
    cards = []
    for index,domain in enumerate(domains):
        prompt = reports[0]['records'][index]['prompt']
        lines += ['### '+domain, '', '**Prompt:** '+prompt, '',
                  '**Referencia del dataset:** '+reports[0]['records'][index]['reference_answer'], '']
        columns=[]
        for report,name in zip(reports,names):
            r=report['records'][index]
            lines += ['#### '+name, '', '````text', r['text'], '````', '',
                      f"Repetición de trigramas: {100*r['metrics']['repeated_trigram_fraction']:.1f} %. Parada: {r['stop_reason']}.", '']
            columns.append('<article><h3>'+html.escape(name)+'</h3><pre>'+html.escape(r['text'])+'</pre></article>')
        cards.append('<h2>'+domain+'</h2><p>'+html.escape(prompt)+'</p><div class="grid">'+''.join(columns)+'</div>')
    lines += ['## Límites de interpretación', '',
              'Es una comprobación cualitativa con tres preguntas, no un benchmark de inteligencia. '
              'No hay calentamiento separado ni repeticiones estadísticas. No se vacía la caché del sistema operativo. '
              'Comparar tiempo total exige mirar también la longitud y la calidad. '
              'Un bloque Python con sintaxis válida puede resolver una tarea distinta: no se ejecuta código generado. '
              'La repetición de trigramas es un indicador léxico, no una puntuación de coherencia. '
              'Comparar estas tres variantes no aísla por sí solo el efecto del finetuning: para eso se requieren '
              'controles restringidos sin adaptar; existen evaluaciones previas, con sus protocolos documentados.', '']
    (directory/'summary.md').write_text('\n'.join(lines),encoding='utf8')
    (directory/'outputs.html').write_text('<!doctype html><meta charset="utf-8"><title>Generación OLMoE</title>'
        '<style>body{font:16px system-ui;margin:30px;background:#f1f5f9;color:#172033}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}'
        'article{padding:20px;background:white;border-radius:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 monospace}'
        'img{width:100%}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>'
        '<h1>OLMoE: textos completos y recursos</h1><p>Original / recuperado 32 / recuperado 16. '
        'Tres preguntas reservadas; máximo 256 tokens, greedy y parada EOS. Los tiempos del original incluyen swapping.</p>'
        '<img src="comparison.png" alt="Tiempos, memoria y repetición">'+''.join(cards),encoding='utf8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    render(parser.parse_args().directory)
