"""Single entry point. Heavy dependencies are loaded only for the chosen command."""
import argparse
import importlib
import sys


COMMANDS = {
    'bank-eval': ('asi.experiments.bank_eval', 'main', 'Evaluar el banco independiente con el protocolo compartido GPT-2'),
    'bank-audit': ('asi.experiments.train_bank', 'audit_main', 'Verificar integridad y finalización de los ocho especialistas'),
    'reference-code-check': ('asi.analysis.reference_code_check', 'main', 'Verificar referencias de código en contenedores Linux aislados'),
    'training-pairing-report': ('asi.analysis.training_pairing', 'main', 'Verificar pesos iniciales y lotes reales entre entrenamientos pareados'),
    'routing-scale': ('asi.experiments.routing_scale', 'main', 'Comparar tres escalas de mezcla con máscara fija en desarrollo revisado'),
    'recovery-data-audit': ('asi.analysis.recovery_data_audit', 'main', 'Inventariar exposición histórica y candidatos sin congelar un nuevo test'),
    'train-bank': ('asi.experiments.train_bank', 'main', 'Entrenar y reanudar el banco de especialistas densos independientes'),
    'mask-intervention-report': ('asi.analysis.mask_intervention_report', 'main', 'Graficar intervenciones de pools, capas libres o escala de mezcla'),
    'mask-intervention': ('asi.experiments.mask_intervention', 'main', 'Recalibrar pools y comparar capas libres con controles de memoria'),
    'routing-paired': ('asi.experiments.routing_paired', 'main', 'Comparar routing sobre tokens idénticos antes y después de recuperación'),
    'routing-paired-report': ('asi.analysis.routing_paired_report', 'main', 'Graficar trazas emparejadas sin cargar modelos'),
    'recovery-pilot-report': ('asi.analysis.recovery_pilot_report', 'main', 'Analizar el piloto completado y sus mapas opcionales t-SNE'),
    'local-generation': ('asi.experiments.local_generation', 'main', 'Comparar memoria y generación de GPT-2, DeepSeek y ASI con el mismo protocolo'),
    'local-generation-report': ('asi.analysis.local_generation_report', 'main', 'Graficar la comparación de generación de modelos locales'),
    'recovery-hardware': ('asi.analysis.recovery_hardware', 'main', 'Calibrar perfiles OLMoE con calidad dev y memoria de inferencia medida'),
    'deepseek-eval': ('asi.experiments.native_eval', 'main', 'Evaluar el DeepSeek original con protocolo GPT-2 y auditar su routing libre'),
    'capacity-compare': ('asi.analysis.capacity_compare', 'main', 'Comparar semillas pareadas y evolución del routing en el piloto de capacidad'),
    'capacity-pilot': ('asi.experiments.capacity_pilot', 'main', 'Piloto pareado de capacidad con ventanas fijas y telemetría de expertos'),
    'conversation-compare': ('asi.experiments.conversation_compare', 'main', 'Comparar GPT-2 y modelo categorizado con el mismo formato y muestreo'),
    'capacity-study': ('asi.experiments.capacity_study', 'main', 'Preparar controles de capacidad y granularidad sin entrenar modelos'),
    'domain-sft': ('asi.experiments.domain_sft', 'main', 'Fine-tuning OASST del modelo categorizado manteniendo sus pools'),
    'checkpoint-report': ('asi.analysis.checkpoint_report', 'main', 'Resumir y graficar benchmarks del checkpoint y GPT-2'),
    'teacher-audit': ('asi.experiments.teacher_audit', 'main', 'Verificar referencias del profesor tras una interrupción'),
    'checkpoint-eval': ('asi.experiments.checkpoint_eval', 'main', 'Comparar checkpoint por pools con los benchmarks del proyecto GPT-2'),
    'recovery-study': ('asi.experiments.recovery_study', 'main', 'Preparar comparación multisemilla y revisión ciega de recuperación'),
    'recovery': ('asi.experiments.recovery', 'main', 'Calibrar selecciones y ajustar routers o expertos LoRA con memoria limitada'),
    'public-moe': ('asi.experiments.public_moe', 'main', 'Preparar y comparar el MoE público OLMoE con AirLLM'),
    'routing-study': ('asi.experiments.routing_study', 'main', 'Separar residencia, activación y pesos de mezcla en OLMoE'),
    'tasks': ('asi.runtime.tasks', 'main', 'Proponer y planificar subtareas con dependencias y límite de cambios de pools'),
    'taxonomy': ('asi.taxonomy', 'main', 'Validar taxonomía común, declarar categorías del modelo y resolver etiquetas'),
    'adaptive': {
        'budgets': ('asi.runtime.adaptive', 'budgets_main', 'Explorar presupuestos GPU con calidad calibrada e identidad de modelo verificada'),
        'run': ('asi.runtime.adaptive', 'main', 'Planificar y ejecutar pools simultáneos con memoria limitada'),
        'evaluate': ('asi.experiments.adaptive', 'main', 'Calibrar degradación de mezclas y expertos INT8'),
        'train-classifier': ('asi.runtime.classifier', 'main', 'Entrenar una cabeza multietiqueta con embeddings congelados'),
    },
    'design': ('asi.models.design', 'main', 'Diseñar pools y arquitecturas bajo un presupuesto exacto de parámetros'),
    'graphs': ('asi.analysis.plots', 'main', 'Explicar y graficar pruebas ya ejecutadas, sin cargar modelos'),
    'comparison': ('asi.experiments.comparison', 'main', 'Preparar y ejecutar por separado la comparación de tres modelos'),
    'data': {
        'recovery-revise': ('asi.data.recovery_corpus', 'revision_main', 'Construir borrador versionado con correcciones y procedencia conservada'),
        'recovery-corpus': ('asi.data.recovery_corpus', 'main', 'Preparar corpus público de recuperación con temas reservados'),
        'classifier-examples': ('asi.data.classifier_examples', 'main', 'Crear un corpus semilla sintético con splits separados'),
        'prepare': ('asi.data.prepare', 'main', 'Clasificar documentos y crear shards'),
        'pools': ('asi.data.pools', 'main', 'Construir el manifiesto de pools'),
        'inspect': ('asi.data.inspect', 'main', 'Consultar la distribución del dataset'),
    },
    'train': ('asi.experiments.train', 'main', 'Entrenar expertos con pools definidos'),
    'audit': ('asi.experiments.audit', 'main', 'Auditar rutas y memoria del modelo por pools'),
    'cache-study': ('asi.experiments.cache_study', 'main', 'Entrenar predictor y comparar cachés en sesiones separadas'),
    'decode-study': ('asi.experiments.decode_study', 'main', 'Medir swapping por token con residencia limitada por capa'),
    'fixed-study': ('asi.experiments.fixed_study', 'main', 'Medir degradación con expertos fijos N sobre N sin E'),
    'export-store': ('asi.runtime.storage', 'main', 'Exportar backbone y expertos para carga SSD/RAM/GPU'),
    'posthoc': {
        'calibrate': ('asi.experiments.posthoc', 'main', 'Clasificar y comparar expertos ya entrenados'),
    },
    'chat': {
        'deepseek': ('asi.experiments.local_generation', 'chat_main', 'Conversar con el DeepSeek original y routing libre'),
        'router': ('asi.runtime.ensemble', 'main', 'Elegir un modelo denso independiente por tema'),
        'domain': ('asi.runtime.generation', 'domain_main', 'Conversar con el modelo entrenado por pools'),
        'existing': ('asi.runtime.generation', 'existing_main', 'Conversar con el modelo original categorizado'),
    },
    'quantize': ('asi.quantization', 'main', 'Exportar expertos INT8; no activa inferencia comprimida'),
    'plan': ('asi.plan', 'main', 'Consultar la correspondencia con los pasos del TFG'),
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    commands = COMMANDS
    prefix = 'python -m asi'
    while isinstance(commands, dict):
        parser = argparse.ArgumentParser(prog=prefix)
        parser.add_argument('command', choices=commands, help=' | '.join(
            f'{key}: {value[2] if isinstance(value, tuple) else "subcomandos"}'
            for key, value in commands.items()))
        # Parse only the command so that leaf parsers own all their flags.
        selected = parser.parse_args(args[:1]).command
        args = args[1:]
        prefix += ' ' + selected
        commands = commands[selected]
    module, function, _ = commands
    previous = sys.argv
    try:
        sys.argv = [prefix, *args]
        getattr(importlib.import_module(module), function)()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()
