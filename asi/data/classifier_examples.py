"""Original synthetic seed data. No scraped material or claim of human review."""
import argparse
from collections import Counter
import json
from pathlib import Path
from asi import ROOT
from asi.taxonomy import Taxonomy


# Each domain has independent train / validation / held-out test scenarios.
# Templates for label anchors are used ONLY in training.
SCENARIOS = {
 'arts_humanities': ['Analiza el uso de la luz en un retrato barroco.', 'How does an unreliable narrator change a novel?', 'Compara la composición de dos esculturas renacentistas.'],
 'business_economics': ['¿Cómo cambia la demanda de un bien cuando sube su precio?', 'Explain the difference between fixed and variable business costs.', 'Una empresa tiene muchos ingresos pero pierde dinero: ¿cómo puede ocurrir?'],
 'computer_science_software_engineering': ['Escribe una función Python que elimine elementos duplicados conservando el orden.', 'Why can two threads updating the same variable lose an update?', '¿Qué índice de base de datos conviene para búsquedas por fecha?'],
 'education_pedagogy': ['Diseña una actividad para enseñar fracciones a una clase de primaria.', 'How can formative assessment guide the next lesson?', '¿Cómo adaptar una explicación para alumnos con conocimientos previos distintos?'],
 'engineering_technology': ['Explica cómo dimensionar conceptualmente una viga para soportar una carga.', 'What does a feedback controller do in a motor?', '¿Cómo se transforma un plano técnico en un proceso de fabricación?'],
 'environmental_science_energy': ['¿Qué consecuencias tiene la pérdida de un humedal para su entorno?', 'Compare renewable electricity generation and energy storage.', '¿Qué datos permiten estudiar cambios de temperatura durante décadas?'],
 'history_geography': ['¿Cómo influyeron las rutas comerciales en las ciudades medievales?', 'Explain how map projections distort area.', 'Compara la organización territorial de dos imperios antiguos.'],
 'language_writing': ['Reescribe este párrafo para que sea más claro sin cambiar su significado.', 'Why do some languages mark grammatical gender?', 'Traduce una expresión idiomática explicando por qué no funciona palabra por palabra.'],
 'law_government': ['Explica la diferencia entre poder legislativo y ejecutivo.', 'What is the purpose of a written constitution?', '¿Qué distingue un procedimiento civil de uno penal en términos generales?'],
 'life_sciences_biology': ['Explica cómo se transmite una variante genética entre generaciones.', 'How does natural selection differ from genetic drift?', '¿Por qué una célula necesita regular qué genes expresa?'],
 'machine_learning_ai': ['¿Cómo se diferencia el sobreajuste de un buen ajuste a los datos?', 'Explain what an attention mechanism does in a language model.', '¿Cómo detectar fuga de información entre entrenamiento y evaluación?'],
 'mathematics_statistics': ['Resuelve un sistema de dos ecuaciones lineales y justifica cada paso.', 'Why is correlation insufficient to establish causation?', 'Demuestra por qué la suma de dos números pares también es par.'],
 'medicine_health': ['Explica para qué sirve la vacunación desde el punto de vista de salud pública.', 'What is the difference between a symptom and a diagnosis?', '¿Qué diferencia hay entre prevención primaria y secundaria?'],
 'personal_finance_practical_life': ['Ayúdame a distribuir un presupuesto doméstico entre gastos y ahorro.', 'Explain the difference between simple and compound interest on savings.', '¿Cómo comparar el coste total de dos préstamos sin elegir uno por mí?'],
 'philosophy_ethics': ['Compara una ética basada en consecuencias con otra basada en deberes.', 'What is the difference between belief and justified knowledge?', '¿Qué significa que un argumento sea válido pero tenga premisas falsas?'],
 'physical_sciences': ['Explica por qué cambia la presión de un gas cuando varía su temperatura.', 'Why do some chemical reactions release heat?', '¿Cómo inferimos la composición de una estrella a partir de su luz?'],
 'social_sciences': ['¿Cómo influyen las normas de un grupo en el comportamiento individual?', 'How can a survey suffer from selection bias in social research?', 'Compara formas de organización familiar en distintos contextos sociales.'],
 'agriculture_food': ['¿Cómo ayuda la rotación de cultivos a conservar el suelo?', 'What factors affect animal welfare in livestock production?', 'Explica la cadena de suministro desde la cosecha hasta el mercado.'],
 'culinary': ['¿Cómo preparar una emulsión estable para una salsa?', 'Why does bread dough need time to rise?', 'Adapta una receta para hornearla en un molde de distinto tamaño.'],
 'sports_recreation': ['Explica cuándo se considera fuera de juego en fútbol.', 'How should a beginner structure a gradual running routine?', '¿Qué material básico se utiliza para practicar orientación al aire libre?'],
 'games': ['Explica qué significa controlar el centro en ajedrez.', 'How can a board game balance luck and player decisions?', 'Propón una mecánica de cooperación para un videojuego.'],
 'religion_spirituality': ['Compara el papel del ayuno en distintas tradiciones religiosas.', 'What is the difference between studying religion and practicing it?', 'Describe cómo se transmiten relatos sagrados en una comunidad.'],
 'crafts_trades': ['¿Cómo elegir una unión de madera para construir una estantería?', 'How can a torn seam in a cotton shirt be repaired?', 'Explica cómo diagnosticar una bisagra que roza antes de reemplazarla.'],
 'home_daily_life': ['Organiza un calendario sencillo para las tareas de una vivienda compartida.', 'How do indoor plants respond to poor drainage?', '¿Qué conviene medir antes de reorganizar los muebles de una habitación?'],
 'relationships_care': ['¿Cómo expresar un desacuerdo sin atacar a la otra persona?', 'Suggest ways to share caregiving responsibilities within a family.', '¿Cómo explicar a un niño un cambio importante de rutina?'],
 'media_information': ['¿Qué comprobaciones ayudan a verificar la procedencia de una noticia?', 'How does a library catalog help retrieve a document?', 'Distingue información, opinión y publicidad en un artículo.'],
 'security_safety': ['¿Cómo organizar un simulacro de evacuación de un edificio?', 'Explain how multifactor authentication reduces account compromise.', 'Identifica medidas preventivas para riesgos habituales de un taller.'],
 'mobility_travel': ['Compara cómo organizar un trayecto con varios medios de transporte.', 'How does a warehouse decide the order for preparing shipments?', '¿Qué información necesito para orientarme con un mapa y una brújula?'],
 'culture_local_knowledge': ['¿Cómo documentar una tradición oral respetando a la comunidad que la conserva?', 'Explain why local ecological knowledge depends on place and experience.', 'Describe maneras de preservar una práctica cultural sin descontextualizarla.'],
 'fiction_speculation': ['Inventa un mundo donde las ciudades puedan desplazarse por el océano.', 'Write a fictional conversation between two rival astronomers on an imaginary planet.', 'Diseña una sociedad fantástica con reglas de magia coherentes.'],
}

MULTI = [
 ('train', 'Explica la regla de la cadena que se utiliza en la retropropagación.', ['mathematics_statistics', 'machine_learning_ai'], 'coupled'),
 ('val', '¿Cómo intervienen los vectores y las matrices en una red neuronal?', ['mathematics_statistics', 'machine_learning_ai'], 'coupled'),
 ('test', 'Relaciona la derivada de una función de pérdida con el aprendizaje de sus pesos.', ['mathematics_statistics', 'machine_learning_ai'], 'coupled'),
 ('train', 'Calcula la media de estos valores y después escribe una función que reproduzca el cálculo.', ['mathematics_statistics', 'computer_science_software_engineering'], 'dependent'),
 ('val', 'Primero deduce una fórmula para la varianza; después implementa esa fórmula en Python.', ['mathematics_statistics', 'computer_science_software_engineering'], 'dependent'),
 ('test', 'Justifica cómo obtener una raíz por bisección y luego programa ese procedimiento.', ['mathematics_statistics', 'computer_science_software_engineering'], 'dependent'),
 ('train', 'Resume la historia de la imprenta. En una sección aparte, explica cómo hacer una salsa.', ['history_geography', 'culinary'], 'independent'),
 ('val', 'Explica una reacción química y, en otro apartado independiente, diseña una actividad escolar de lectura.', ['physical_sciences', 'education_pedagogy'], 'independent'),
 ('test', 'Describe un ecosistema costero y, por separado, compara dos sistemas electorales.', ['environmental_science_energy', 'law_government'], 'independent'),
 ('train', 'Compara estos algoritmos Python y programa el que requiera menos operaciones.', ['computer_science_software_engineering'], 'dependent'),
 ('val', 'Revisa estas dos funciones y reescribe la que evita recorrer la lista varias veces.', ['computer_science_software_engineering'], 'dependent'),
 ('test', 'Escoge entre estas implementaciones de búsqueda y entrega una versión con nombres más claros.', ['computer_science_software_engineering'], 'dependent'),
]


def generate(taxonomy):
    rows = []
    def add(identifier, text, labels, split, family, **extra):
        rows.append({'id': identifier, 'text': text, 'labels': labels, 'split': split,
                     'family': family, 'origin': 'original_synthetic_seed', 'human_reviewed': False, **extra})
    for node in taxonomy.payload['nodes']:
        if node['id'] in ('knowledge', 'unknown'):
            continue
        key = node['id']
        add('anchor_' + key, f"Quiero aprender sobre {node['name']}. Explica sus conceptos principales.", [key], 'train', 'definition_anchors')
        add('scope_' + key, f"Tema de consulta: {node['name']}. {node['description']}", [key], 'train', 'definition_anchors')
    for key, texts in SCENARIOS.items():
        for split, text in zip(('train', 'val', 'test'), texts):
            add(f'{split}_{key}', text, [key], split, f'{key}_{split}_scenario')
    for i, (split, text, labels, relation) in enumerate(MULTI):
        add(f'multi_{i}', text, labels, split, f'multi_case_{i}', task_relation=relation)
    for i, (text, split) in enumerate([
        ('Haz eso con lo anterior.', 'train'), ('No sé a qué me refiero; ¿puedes adivinarlo?', 'train'),
        ('xqz 884 %%%', 'train'), ('Clasifica el asunto sin que yo te diga cuál es.', 'train'),
        ('¿Y aquello?', 'val'), ('Do the thing I forgot to describe.', 'val'),
        ('Necesito ayuda con…', 'test'), ('Which one? There is no attached example.', 'test')]):
        add(f'unknown_{i}', text, ['unknown'], split, f'unknown_{i}')
    contrasts = [
        ('Describe cómo se alimenta una pitón en su hábitat.', 'life_sciences_biology'),
        ('Describe cómo se ejecuta un programa escrito en Python.', 'computer_science_software_engineering'),
        ('¿Qué función cumple una red eléctrica?', 'engineering_technology'),
        ('¿Qué función cumple una red neuronal?', 'machine_learning_ai'),
        ('Explica qué significa capital en el balance de una empresa.', 'business_economics'),
        ('Explica por qué una ciudad puede convertirse en capital de un territorio.', 'history_geography')]
    for i, (text, label) in enumerate(contrasts):
        add(f'contrast_{i}', text, [label], 'train', 'lexical_contrasts')
    return rows


def audit(rows, taxonomy):
    import tiktoken
    encoder = tiktoken.get_encoding('gpt2')
    texts, families = {}, {}
    counts = {split: Counter() for split in ('train', 'val', 'test')}
    for row in rows:
        if row['split'] not in counts or not row['labels'] or any(label not in taxonomy.labels for label in row['labels']):
            raise ValueError('Invalid example labels or split')
        tokens = tuple(encoder.encode(row['text'])[-512:])
        key = row['text'].casefold().strip()
        for value in (tokens, key):
            if value in texts and texts[value] != row['split']:
                raise ValueError('Cross-split duplicate input')
            texts[value] = row['split']
        if row['family'] in families and families[row['family']] != row['split']:
            raise ValueError('Template family spans splits')
        families[row['family']] = row['split']
        counts[row['split']].update(row['labels'])
    return {'taxonomy': taxonomy.identity, 'examples': len(rows),
            'split_counts': dict(Counter(row['split'] for row in rows)),
            'direct_positive_counts': {split: dict(counter) for split, counter in counts.items()},
            'labels_without_direct_validation': [label for label in taxonomy.labels if not counts['val'][label]],
            'limitations': ['Original synthetic seed, not a representative or human-reviewed training corpus.',
                            'Fine-label training mostly consists of definition anchors. Fine-label generalization is unmeasured.',
                            'Validation/test scenarios are small, predominantly broad labels; do not compare ancestor-inclusive metrics as fine-category accuracy.',
                            'Task-relation metadata is not trained by the topic classifier.',
                            'No exact effective-input or template-family overlap across splits; semantic leakage still needs manual review.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--taxonomy', type=Path, default=ROOT / 'configs/knowledge_v1.json')
    parser.add_argument('--output', type=Path, required=True, help='New directory')
    args = parser.parse_args()
    taxonomy = Taxonomy.read(args.taxonomy)
    rows = generate(taxonomy)
    report = audit(rows, taxonomy)
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, splits in [('train_val.jsonl', {'train', 'val'}), ('heldout_test.jsonl', {'test'})]:
        (args.output / filename).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows if row['split'] in splits), encoding='utf-8')
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'examples': len(rows), 'splits': report['split_counts'], 'output': str(args.output)}, indent=2))
