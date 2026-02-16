# Taxonomia de Clasificacion Jerarquica para Contextos de Memory Cards

Este documento define la estructura jerarquica para clasificar mensajes de usuarios en contextos, entidades y sub-entidades de memory cards, para la extraccion y almacenamiento de informacion.

---

## Estructura de la Taxonomia

**Formato**: `contexto > entidad > sub_entidad`

**Ejemplo**: "Mi objetivo es ganar 120k$ el proximo ano" → `professional > professional_aspirations > compensation_expectations`

**Nota**: El contexto "aspirational" ha sido absorbido dentro de `professional_aspirations` (para metas de carrera) y `personal_goals` (para metas de vida).

---

## 1. Contexto Profesional (`professional`)

### 1.1 `current_position` — Posicion actual
- **role** — Titulo/puesto actual
- **company** — Empresa actual
- **compensation** — Salario actual y compensacion total
- **start_date** — Cuando empezo en el puesto actual

**Ejemplos de mensajes:**
- "Actualmente soy Senior Product Manager en Google"
- "Gano 150k$ de salario base"
- "Llevo 2 anos en este puesto"

### 1.2 `professional_experience` — Experiencia profesional
- **past_roles** — Puestos anteriores
- **past_companies** — Empresas anteriores
- **responsibilities** — Responsabilidades en puestos anteriores
- **achievements** — Logros e impacto
- **duration** — Duracion en cada puesto

**Ejemplos de mensajes:**
- "Trabaje en Microsoft 3 anos como PM"
- "En mi ultimo trabajo lance un feature que aumento los ingresos un 20%"
- "Lideré un equipo de 5 ingenieros"

### 1.3 `awards` — Premios profesionales
- **awards** — Premios y reconocimientos profesionales de empleadores o la industria

**Ejemplos de mensajes:**
- "Gane el premio Product Manager del Ano"
- "Recibi Empleado del Mes en mi empresa"
- "Fui reconocido como Top Performer en mi division"

### 1.4 `licenses_and_permits` — Licencias y permisos
- **driving_licenses** — Licencias de conducir
- **professional_licenses** — Licencias profesionales (medica, abogacia, CPA)
- **security_clearances** — Habilitaciones de seguridad gubernamental
- **operating_permits** — Otros permisos

**Ejemplos de mensajes:**
- "Tengo licencia de conducir de California"
- "Tengo habilitacion de seguridad nivel Secret"
- "Mi licencia medica caduca el proximo ano"

### 1.5 `professional_aspirations` — Aspiraciones profesionales

#### 1.5.1 `dream_roles` — Roles sonados (consolidado: incluye empresas e industrias objetivo)
- **desired_roles** — Roles que quieren alcanzar
- **target_companies** — Empresas especificas donde quieren trabajar
- **target_industries** — Industrias donde quieren trabajar
- **priority** — Que tan importante es este rol
- **timeframe** — Cuando quieren lograrlo
- **readiness** — Que tan preparados se sienten
- **skill_gaps** — Habilidades que necesitan desarrollar
- **connections** — Personas que conocen en empresas objetivo
- **networking_actions** — Pasos que estan tomando

**Ejemplos de mensajes:**
- "Quiero ser VP de Producto en 2 anos"
- "Mi sueno es ser CPO algun dia"
- "Me encantaria trabajar en Stripe" ← empresa objetivo
- "Tengo un amigo en Airbnb que podria recomendarme" ← conexiones
- "Me interesa mucho el fintech" ← industria objetivo
- "Quiero ser Senior PM en Google en la industria fintech" ← rol + empresa + industria

**Razon de la consolidacion:**
En conversacion natural, la gente a menudo expresa roles sonados, empresas e industrias objetivo juntos (ej: "Quiero ser Senior PM en Google en fintech"). Separarlos creaba fronteras artificiales. Esta consolidacion refleja mejor como la gente habla naturalmente de sus aspiraciones.

#### 1.5.2 `compensation_expectations` — Expectativas de compensacion
- **target_salary** — Salario deseado
- **minimum_acceptable** — Minimo aceptable
- **stretch_goal** — Objetivo aspiracional
- **total_comp** — Incluyendo equity/bonus
- **flexibility** — Que tan flexibles son
- **priorities** — Que mas importa (equity, beneficios)

**Ejemplos de mensajes:**
- "Mi objetivo es ganar 120k$ el proximo ano"
- "Necesito al menos 180k$ de base"
- "Quiero 350k$ de compensacion total con equity"
- "Soy flexible con el salario si el equity es bueno"

#### 1.5.3 `desired_work_environment` — Entorno de trabajo deseado
- **work_mode** — Remoto/hibrido/presencial
- **company_size** — Startup/scale-up/corporacion
- **company_stage** — Seed/Series A/publica
- **management** — Contribuidor individual vs manager
- **culture_priorities** — Atributos culturales que buscan
- **deal_breakers** — Cosas innegociables

**Ejemplos de mensajes:**
- "Quiero trabajar hibrido, no completamente remoto"
- "Busco una startup en Serie C"
- "Necesito equilibrio vida-trabajo"
- "No trabajare en un sitio con guardias nocturnas"

#### 1.5.4 `career_change_considerations` — Consideraciones de cambio de carrera
- **considering_change** — Pensando en un cambio de carrera
- **change_type** — De rol/industria/ambos
- **risk_tolerance** — Disposicion a tomar riesgos
- **pay_cut** — Aceptarian menor salario
- **obstacles** — Que les frena
- **support_needed** — Que ayuda necesitan

**Ejemplos de mensajes:**
- "Estoy pensando en cambiar de ingenieria a PM"
- "Quiero cambiar de industria pero me preocupa la bajada de salario"
- "Necesito mas presencia ejecutiva para el puesto de VP"

#### 1.5.5 `job_search_status` — Estado de busqueda de empleo
- **currently_searching** — Buscando activamente o no
- **urgency** — Que tan urgente es la busqueda
- **applications** — Cuantas candidaturas enviadas
- **interviews** — Entrevistas en proceso
- **offers** — Ofertas recibidas
- **start_date** — Cuando quieren empezar

**Ejemplos de mensajes:**
- "Estoy mirando oportunidades de forma casual"
- "He aplicado a 5 empresas"
- "Tengo 2 entrevistas la proxima semana"
- "He recibido una oferta de Google"

### 1.6 `volunteer_experience` — Experiencia de voluntariado
- **volunteer_roles** — Posiciones de voluntariado

**Ejemplos de mensajes:**
- "Soy mentor de PMs junior en Product School"

---

## 2. Contexto de Aprendizaje (`learning`)

### 2.1 `current_skills` — Habilidades actuales
- **skills** — Habilidades que tienen actualmente
- **proficiency** — Nivel de competencia
- **experience** — Anos de experiencia
- **verification** — Como se verifica la habilidad

**Ejemplos de mensajes:**
- "Soy experto en estrategia de producto"
- "Se Python a nivel intermedio"
- "Tengo 8 anos de experiencia en analisis de datos"

### 2.2 `languages` — Idiomas
- **language** — Idiomas que hablan
- **proficiency** — Nivel de competencia
- **certifications** — Puntuaciones/certificaciones de tests de idiomas

**Ejemplos de mensajes:**
- "Hablo espanol a nivel B1"
- "Soy bilingue en mandarin"
- "Saque un 110 en el TOEFL"

### 2.3 `education_history` — Historial educativo
- **degrees** — Titulos obtenidos
- **institutions** — Instituciones donde estudiaron
- **field_of_study** — Carrera/especializacion
- **gpa** — Rendimiento academico
- **graduation_date** — Fecha de graduacion

**Ejemplos de mensajes:**
- "Tengo un Grado en Informatica por la Universidad de Berkeley"
- "Me gradue con un 3.7 de GPA"
- "Estudie Ingenieria de Software"

### 2.4 `learning_gaps` — Carencias de aprendizaje (agrupado: habilidades y conocimiento)

Habilidades y conocimientos que faltan y bloquean objetivos de carrera.

#### 2.4.1 `skill_gaps` — Carencias de habilidades
- **missing_skills** — Habilidades que faltan para alcanzar metas
- **impact** — Como afecta esta carencia
- **blocking_aspiration** — El rol, habilidad o meta especifica que bloquea
- **aspiration_type** — Si bloquea un rol, habilidad, certificacion o meta general

**Ejemplos de mensajes:**
- "Necesito mejorar mi presencia ejecutiva para ser VP"
- "Me falta experiencia gestionando personas para el puesto de Director"
- "No tengo suficientes habilidades tecnicas para ese puesto de Senior Engineer"
- "Me falta experiencia en Python para pasar a data science"

#### 2.4.2 `knowledge_gaps` — Carencias de conocimiento
- **missing_knowledge** — Conocimiento que necesitan desarrollar
- **blocking_aspiration** — El rol, habilidad o meta especifica que bloquea
- **aspiration_type** — Si bloquea un rol, habilidad, certificacion o meta general

**Ejemplos de mensajes:**
- "No entiendo blockchain lo suficiente para trabajar en Coinbase"
- "Necesito aprender mas sobre IA/ML para pasarme a machine learning"
- "Me falta conocimiento de fintech para el puesto de pagos que quiero"

**Ejemplo combinado:**
- "Necesito mejor presencia ejecutiva y conocimiento mas profundo de fintech para ser VP en una empresa de pagos"

**Razon de la agrupacion:**
Tanto las carencias de habilidades como las de conocimiento representan competencias que faltan y bloquean aspiraciones. Se mencionan juntas frecuentemente y sirven el mismo proposito: identificar que hay que aprender para alcanzar metas.

### 2.5 `learning_aspirations` — Aspiraciones de aprendizaje (agrupado: habilidades, educacion y certificaciones)

Metas futuras de aprendizaje en todos los dominios.

#### 2.5.1 `skill_aspirations` — Aspiraciones de habilidades
- **target_skills** — Habilidades que quieren aprender
- **learning_plan** — Como planean aprender
- **timeline** — Cuando quieren aprenderlo
- **progress** — Progreso actual

**Ejemplos de mensajes:**
- "Quiero aprender machine learning"
- "Estoy aprendiendo a hablar en publico en Toastmasters"
- "Llevo un 25% de mi curso de IA"

#### 2.5.2 `education_aspirations` — Aspiraciones educativas
- **desired_degrees** — Titulos que quieren obtener
- **institutions** — Universidades objetivo
- **timeline** — Cuando planean estudiar
- **funding** — Como lo financiaran

**Ejemplos de mensajes:**
- "Quiero hacer un MBA en Stanford"
- "Planeo hacer un master en IA"
- "Aplicare a escuelas de negocios en 2027"

#### 2.5.3 `certification_aspirations` — Aspiraciones de certificacion
- **target_certs** — Certificaciones que quieren
- **study_plan** — Como se estan preparando
- **exam_date** — Cuando planean examinarse

**Ejemplos de mensajes:**
- "Estoy estudiando para la certificacion de Google Cloud"
- "Quiero sacar el PMP el proximo ano"
- "Estoy haciendo un curso preparatorio para AWS"

**Ejemplo combinado:**
- "Quiero aprender Python y sacar la certificacion AWS mientras hago un MBA"

**Razon de la agrupacion:**
Los tres tipos de aspiraciones de aprendizaje (habilidades, educacion, certificaciones) son metas de aprendizaje orientadas al futuro que se solapan naturalmente y se expresan juntas frecuentemente.

### 2.6 `certifications` — Certificaciones obtenidas
- **earned_certs** — Certificaciones que tienen
- **issue_date** — Cuando la obtuvieron
- **expiry_date** — Cuando caduca
- **status** — Activa/caducada

**Ejemplos de mensajes:**
- "Tengo la certificacion AWS Solutions Architect"
- "Tengo mi certificacion CSPO"
- "Mi PMP caduca el proximo ano"

### 2.7 `knowledge_areas` — Areas de conocimiento
- **expertise_domains** — Areas de conocimiento amplias

**Ejemplos de mensajes:**
- "Tengo conocimientos profundos en fintech y pagos"
- "Tengo experiencia profunda en arquitectura de plataformas"

### 2.8 `learning_preferences` — Preferencias de aprendizaje
- **preferred_formats** — Como les gusta aprender
- **pace** — Rapido/lento aprendiendo
- **budget** — Presupuesto para aprendizaje
- **time_available** — Horas por semana para aprender

**Ejemplos de mensajes:**
- "Aprendo mejor con proyectos practicos"
- "Prefiero libros a videos"
- "Puedo dedicar 10 horas por semana a aprender"
- "Tengo 2000$/ano para cursos"

### 2.9 `learning_history` — Historial de aprendizaje
- **past_courses** — Cursos realizados
- **books** — Libros leidos
- **outcomes** — Que aprendieron

**Ejemplos de mensajes:**
- "Hice el curso de ML de Andrew Ng en Coursera"
- "Lei 'High Output Management'"
- "Complete un bootcamp de data science"

### 2.10 `publications` — Publicaciones
- **publications** — Articulos, papers, posts escritos

**Ejemplos de mensajes:**
- "Escribi un articulo sobre productos de plataforma"
- "Publique un paper sobre machine learning"
- "Mantengo un blog tecnico sobre IA"

### 2.11 `academic_awards` — Premios academicos
- **academic_awards** — Honores y reconocimientos academicos de instituciones educativas

**Ejemplos de mensajes:**
- "Estuve en la Lista del Decano en la universidad"
- "Me gradue Summa Cum Laude"
- "Recibi el premio al Estudiante Sobresaliente"

---

## 3. Contexto Social (`social`)

### 3.1 `mentors` — Mentores
- **mentor_name** — Nombre del mentor
- **mentor_role** — Su rol/titulo
- **relationship** — Formal/informal
- **frequency** — Con que frecuencia se reunen
- **guidance_areas** — En que les ayudan
- **impact** — Que tan util es

**Ejemplos de mensajes:**
- "Tengo un mentor que es VP en Stripe"
- "Mi mentor se reune conmigo mensualmente"
- "Sarah me ayuda con habilidades de liderazgo"

### 3.2 `mentees` — Mentorados
- **mentee_name** — Nombre del mentorado
- **mentee_background** — Su background
- **guidance_provided** — Que ayuda se proporciona
- **progress** — Como estan progresando

**Ejemplos de mensajes:**
- "Soy mentor de un PM junior que viene de diseno"
- "Ayudo a PMs nuevos con estrategia de producto"

### 3.3 `professional_network` — Red profesional
- **connections** — Personas en su red (pares, colegas, conocidos)
- **relationship_strength** — Vinculos fuertes/debiles
- **interaction_frequency** — Con que frecuencia interactuan
- **collaboration_type** — Como trabajan juntos
- **last_interaction** — Cuando fue la ultima interaccion
- **communities** — Comunidades y grupos profesionales
- **community_type** — Online/presencial
- **membership_status** — Miembro activo/inactivo
- **engagement_level** — Nivel de participacion
- **community_value** — Que obtienen de las comunidades

**Ejemplos de mensajes:**
- "Tengo un companero de accountability para metas de carrera"
- "Colaboro con otros PMs en mi cohorte de Reforge"
- "Conozco al Director de Producto de Airbnb"
- "Tengo 450 conexiones en LinkedIn"
- "Estoy en la comunidad de Reforge"
- "Asisto a meetups de Product Managers en SF"

### 3.4 `recommendations` — Recomendaciones (consolidado: testimonios y referencias)
- **testimonial_from** — Quien escribio el testimonio
- **testimonial_text** — La recomendacion escrita
- **permission_to_share** — Se puede compartir publicamente
- **reference_name** — Nombre de la referencia
- **reference_role** — Su titulo/posicion
- **relationship** — Como se conocen

**Ejemplos de mensajes:**
- "Mi manager escribio una buena recomendacion en LinkedIn"
- "Tengo un testimonio de un colega"
- "John Smith puede ser referencia para mi"
- "Puedo dar 3 referencias profesionales"

**Razon de la consolidacion:**
Testimonios (respaldos escritos) y referencias (personas que avalan) son dos caras de la misma moneda — ambos son validaciones de otros sobre tus habilidades y experiencia.

### 3.5 `networking` — Networking (agrupado: actividades, metas y preferencias)

Actividades, metas y preferencias de networking profesional.

#### 3.5.1 `networking_activities` — Actividades de networking
- **activity_type** — Conferencia/cafe/etc.
- **date** — Cuando ocurrio
- **people_met** — A quien conocieron
- **follow_up** — Proximos pasos

**Ejemplos de mensajes:**
- "Asisti a ProductCon el mes pasado"
- "Tome un cafe con un PM de Google"
- "Voy a Stripe Sessions en marzo"

#### 3.5.2 `networking_goals` — Metas de networking
- **target_connections** — Personas que quieren conocer
- **target_events** — Eventos a los que quieren asistir
- **networking_strategy** — Como haran networking

**Ejemplos de mensajes:**
- "Quiero conocer mas CPOs"
- "Planeo asistir a 3 conferencias este ano"

#### 3.5.3 `networking_preferences` — Preferencias de networking
- **preferred_formats** — 1-a-1/grupos/conferencias
- **energy_impact** — Energizante/agotador
- **style** — Enfoque al networking

**Ejemplos de mensajes:**
- "Prefiero cafes 1-a-1 a grandes eventos"
- "El networking me agota, necesito tiempo de recuperacion"

---

## 4. Contexto Psicologico (`psychological`)

### 4.1 `personality_profile` — Perfil de personalidad
- **personality_type** — MBTI, Big Five, etc.
- **traits** — Rasgos clave de personalidad
- **self_description** — Como se describen a si mismos

**Ejemplos de mensajes:**
- "Soy INTJ"
- "Soy introvertido y analitico"
- "Soy perfeccionista"

### 4.2 `values` — Valores profesionales
- **professional_values** — Que importa en el trabajo
- **priorities** — Prioridades de valores

**Ejemplos de mensajes:**
- "Valoro la autonomia y el impacto"
- "El equilibrio vida-trabajo es mi principal prioridad"
- "Me importa mucho la alineacion con la mision"

### 4.3 `motivations` — Motivaciones
- **intrinsic_motivations** — Motivadores internos
- **extrinsic_motivations** — Motivadores externos
- **demotivators** — Que les desmotiva

**Ejemplos de mensajes:**
- "Me motiva resolver problemas dificiles"
- "El dinero no es mi principal motivador"
- "La microgestion mata mi motivacion"

### 4.4 `working_style_preferences` — Preferencias de estilo de trabajo
- **work_style** — Como prefieren trabajar
- **collaboration_style** — Como trabajan con otros
- **decision_making** — Como toman decisiones
- **communication_style** — Como se comunican

**Ejemplos de mensajes:**
- "Trabajo mejor de forma independiente con objetivos claros"
- "Me gustan las sesiones de brainstorming colaborativo"
- "Tomo decisiones rapido basandome en datos"

### 4.5 `confidence_and_self_perception` — Confianza y auto-percepcion

#### 4.5.1 `confidence_levels` — Niveles de confianza
- **overall_confidence** — Nivel de confianza general
- **confidence_changes** — Como ha cambiado recientemente
- **domain_confidence** — Confianza por dominio (tecnico, social, liderazgo, hablar en publico, decisiones de carrera)
- **confidence_factors** — Que afecta la confianza

**Ejemplos de mensajes:**
- "Me siento bastante seguro ultimamente (7/10)"
- "Mi confianza ha bajado despues del fracaso del proyecto"
- "Soy muy seguro tecnicamente (8/10)"
- "Me cuesta hablar en publico (4/10)"

#### 4.5.2 `imposter_syndrome_and_doubt` — Sindrome del impostor y dudas
- **imposter_level** — Que tan fuertes son los sentimientos de impostor
- **imposter_frequency** — Con que frecuencia lo sienten
- **imposter_triggers** — Que dispara los sentimientos de impostor
- **self_doubt_frequency** — Con que frecuencia dudan de si mismos
- **doubt_situations** — Cuando aparecen las dudas
- **comparison_patterns** — Con que frecuencia se comparan con otros
- **self_efficacy** — Creencia en su capacidad de triunfar y crecer
- **resilience** — Capacidad de recuperarse de contratiempos

**Ejemplos de mensajes:**
- "Tengo sindrome del impostor moderado"
- "Me siento un fraude cuando presento a lideres senior"
- "A menudo dudo si soy lo suficientemente listo"
- "Creo que puedo aprender cualquier cosa con esfuerzo"
- "Me recupero rapido de los contratiempos"

#### 4.5.3 `self_talk_and_validation` — Dialogo interno y validacion
- **inner_critic_strength** — Que tan duro es el dialogo interno
- **self_compassion** — Que tan amables son consigo mismos
- **negative_thought_patterns** — Pensamientos negativos comunes
- **external_validation_need** — Necesidad de aprobacion externa
- **internal_validation_ability** — Capacidad de auto-validarse
- **reaction_to_criticism** — Como manejan las criticas
- **reaction_to_praise** — Como manejan los elogios

**Ejemplos de mensajes:**
- "Mi critico interno es muy duro"
- "Me castigo por los errores"
- "Estoy aprendiendo a ser mas compasivo conmigo mismo"
- "Dependo demasiado de la aprobacion de otros"
- "Rechazo los cumplidos"

#### 4.5.4 `confidence_building_strategies` — Estrategias de construccion de confianza
- **strategies_that_help** — Que construye confianza
- **strategies_that_hurt** — Que dana la confianza
- **currently_working_on** — Esfuerzos actuales para construir confianza
- **confidence_goals** — Metas de confianza
- **coping_strategies** — Como lidian con la baja confianza

**Ejemplos de mensajes:**
- "Llevar un diario de logros ayuda a mi confianza"
- "Trabajo con un coach ejecutivo en confianza"
- "Mi meta es llegar a 8/10 de confianza"
- "Hablar con mi mentor ayuda con los sentimientos de impostor"

### 4.6 `career_decision_making_style` — Estilo de toma de decisiones de carrera
- **decision_style** — Analitico/intuitivo/etc.
- **decision_factors** — Que influye en las decisiones
- **decision_confidence** — Confianza en las decisiones

**Ejemplos de mensajes:**
- "Tomo decisiones de carrera basandome en datos"
- "Confio en mi instinto al elegir trabajos"
- "Me cuesta tomar grandes decisiones de carrera"

### 4.7 `work_environment_preferences` — Preferencias de entorno laboral
- **ideal_environment** — En que entorno prosperan
- **stressors** — Que les estresa en el trabajo
- **energizers** — Que les da energia

**Ejemplos de mensajes:**
- "Prospero en entornos de ritmo rapido"
- "Las oficinas abiertas me estresan"
- "Me encantan los proyectos colaborativos"

### 4.8 `stress_and_coping` — Estres y afrontamiento
- **stress_level** — Nivel de estres actual
- **stress_triggers** — Que causa estres
- **coping_strategies** — Como lo manejan
- **effectiveness** — Que funciona/no funciona

**Ejemplos de mensajes:**
- "Estoy bastante estresado ahora mismo (7/10)"
- "Los plazos ajustados me estresan"
- "El ejercicio me ayuda a gestionar el estres"

### 4.9 `emotional_intelligence` — Inteligencia emocional
- **self_awareness** — Comprension de sus propias emociones
- **empathy** — Comprension de las emociones ajenas
- **emotional_regulation** — Gestion de emociones

**Ejemplos de mensajes:**
- "Soy muy consciente de mis emociones"
- "Estoy trabajando en ser mas empatico"

### 4.10 `growth_mindset` — Mentalidad de crecimiento
- **mindset_level** — Mentalidad fija vs de crecimiento
- **beliefs_about_talent** — Innato vs desarrollado
- **approach_to_challenges** — Como abordan los desafios

**Ejemplos de mensajes:**
- "Creo que las habilidades se pueden desarrollar con esfuerzo"
- "Veo los fracasos como oportunidades de aprendizaje"
- "Me encanta desafiarme a mi mismo"

---

## 5. Contexto Personal (`personal`)

### 5.1 `personal_life` — Vida personal
- **life_stage** — Etapa de vida (inicio de carrera, mitad de carrera, asentandose, etc.)
- **age_range** — Rango de edad
- **relationship_status** — Soltero/casado/en pareja/divorciado
- **partner** — Situacion y carrera de la pareja
- **children** — Hijos y edades
- **dependents** — Otros dependientes (padres, familiares)
- **childcare** — Arreglos de cuidado infantil
- **family_support** — Sistema de apoyo (suegros, familiares, amigos)
- **life_transitions** — Transiciones recientes o proximas (boda, divorcio, hijos, nido vacio)
- **life_priorities** — Prioridades actuales de vida (tiempo en familia, enfoque en carrera, equilibrio)

**Ejemplos de mensajes:**
- "Tengo treinta y pocos con una familia joven"
- "Acabo de casarme"
- "Ahora mismo priorizo el tiempo en familia"
- "Estoy casado con un hijo de 1.5 anos"
- "Mi pareja es profesora"
- "Mis suegros ayudan con el cuidado de los ninos"

### 5.2 `health_and_wellbeing` — Salud y bienestar

#### 5.2.1 `physical_health` — Salud fisica
- **overall_health** — Estado general de salud
- **chronic_conditions** — Problemas de salud cronicos
- **energy_levels** — Energia/fatiga
- **limitations** — Limitaciones fisicas

**Ejemplos de mensajes:**
- "Tengo un problema cronico de espalda"
- "Estoy privado de sueno por ser padre primerizo"
- "Tengo poca energia la mayoria de los dias"

#### 5.2.2 `mental_health` — Salud mental
- **conditions** — Condiciones de salud mental
- **severity** — Que tan severas son
- **treatment** — Estado del tratamiento
- **impact_on_work** — Como afecta al trabajo

**Ejemplos de mensajes:**
- "Tengo ansiedad y esta controlada con terapia"
- "Lucho con la depresion"
- "Estoy medicado para el TDAH"
- "El burnout esta afectando mi rendimiento laboral"

#### 5.2.3 `addictions_or_recovery` — Adicciones o recuperacion
- **addiction_type** — Tipo de adiccion
- **status** — Activa/en recuperacion
- **clean_since** — Cuanto tiempo limpio/sobrio
- **recovery_program** — AA/NA/etc.
- **support_system** — Red de apoyo
- **triggers** — Que evitar
- **impact_on_career** — Implicaciones en la carrera

**Ejemplos de mensajes:**
- "Llevo 9 meses sobrio del alcohol"
- "Asisto a reuniones de AA 3 veces por semana"
- "No puedo ir a happy hours despues del trabajo"
- "Mi sobriedad es mi principal prioridad"

#### 5.2.4 `overall_wellbeing` — Bienestar general
- **stress_level** — Estres actual
- **wellbeing_score** — Bienestar general

**Ejemplos de mensajes:**
- "Me siento bastante bien en general (7/10)"
- "Mi nivel de estres es alto ahora mismo"

### 5.3 `living_situation` — Situacion de vivienda
- **housing_type** — Propio/alquiler/etc.
- **location** — Donde viven
- **living_with** — Con quien viven
- **relocation_openness** — Disposicion a mudarse
- **constraints** — Que impide mudarse
- **home_office** — Configuracion para trabajo remoto

**Ejemplos de mensajes:**
- "Tengo casa propia en Austin"
- "No puedo mudarme porque mi pareja tiene plaza fija"
- "Tengo un buen despacho en casa"

### 5.4 `financial_situation` — Situacion financiera
- **stability** — Estabilidad financiera
- **debt** — Situacion de deuda
- **emergency_fund** — Colchon de ahorro
- **dependents** — Dependientes financieros
- **income_dependency** — Ingreso unico/doble
- **risk_tolerance** — Tolerancia al riesgo financiero
- **stress_level** — Estres financiero

**Ejemplos de mensajes:**
- "Tengo 45k$ en prestamos estudiantiles"
- "No puedo permitirme riesgos de carrera ahora mismo"
- "Tengo 3-6 meses de fondo de emergencia"
- "Estoy estresado financieramente por la hipoteca"

### 5.5 `personal_goals` — Metas personales
- **non_career_goals** — Metas de vida no profesionales
- **category** — Salud/familia/relaciones/etc.
- **priority** — Nivel de importancia
- **timeframe** — Cuando quieren lograrlo
- **progress** — Progreso actual

**Ejemplos de mensajes:**
- "Quiero mantener la sobriedad (maxima prioridad)"
- "Quiero perder 10 kilos en 6 meses"
- "Quiero estar mas presente con mi familia"
- "Quiero llevar a mi pareja de viaje de aniversario"

### 5.6 `personal_projects` — Proyectos personales
- **project_name** — Nombre del proyecto (tanto relevantes para la carrera como hobbies)
- **project_description** — Que hace el proyecto
- **project_type** — Relacionado con carrera/hobby/creativo/etc.
- **project_role** — Su rol en el proyecto
- **project_skills** — Habilidades usadas (si aplica)
- **time_commitment** — Horas por semana
- **motivation** — Por que lo hacen

**Ejemplos de mensajes:**
- "Construi un dashboard de analiticas open-source"
- "Mantengo un blog de product management con 10k lectores"
- "Estoy restaurando una moto vintage con mi padre"
- "Tengo un huerto en el jardin"

### 5.7 `lifestyle_preferences` — Preferencias de estilo de vida
- **work_life_balance** — Que tan importante es
- **ideal_schedule** — Horario de trabajo preferido
- **flexibility_needs** — Que flexibilidad necesitan
- **non_negotiables** — Que no negociaran

**Ejemplos de mensajes:**
- "El equilibrio vida-trabajo es critico (10/10 de importancia)"
- "Necesito flexibilidad para las reuniones de AA"
- "No trabajare mas de 45 horas por semana"
- "El trabajo remoto es innegociable"

### 5.8 `life_constraints` — Restricciones de vida
- **constraint_type** — Familia/salud/ubicacion/financiera
- **description** — Que es la restriccion
- **impact_on_career** — Como afecta la carrera
- **severity** — Que tan limitante es
- **timeframe** — Cuanto tiempo durara

**Ejemplos de mensajes:**
- "No puedo viajar por responsabilidades de cuidado infantil"
- "Necesito quedarme cerca de mi madre por apoyo medico"
- "No puedo permitirme una bajada de salario"
- "Mis reuniones de recuperacion limitan mi disponibilidad por las tardes"

### 5.9 `life_enablers` — Facilitadores de vida
- **enabler_type** — Familia/apoyo/ubicacion/etc.
- **description** — Que les ayuda
- **benefit_to_career** — Como ayuda a la carrera
- **strength** — Que tan fuerte es el facilitador

**Ejemplos de mensajes:**
- "Mis suegros cuidan a los ninos gratis"
- "Mi pareja apoya mucho mi carrera"
- "Mi comunidad de AA me mantiene responsable"

### 5.10 `major_life_events` — Eventos importantes de vida
- **event_type** — Boda/nacimiento/mudanza/salud/etc.
- **date** — Cuando ocurrio
- **description** — Que paso
- **impact** — Como les afecto

**Ejemplos de mensajes:**
- "Me case el ano pasado"
- "Mi primer hijo nacio en 2023"
- "Empece la recuperacion hace 9 meses"
- "Compramos nuestra primera casa"

### 5.11 `personal_values` — Valores personales
- **life_values** — Que importa en la vida
- **importance** — Nivel de prioridad

**Ejemplos de mensajes:**
- "La familia es mi principal prioridad"
- "La salud y la sobriedad son lo mas importante"
- "Valoro la autenticidad y la honestidad"

### 5.12 `life_satisfaction` — Satisfaccion vital
- **overall_satisfaction** — Satisfaccion general con la vida
- **satisfaction_by_area** — Desglose por areas
- **areas_to_improve** — Que quieren mejorar

**Ejemplos de mensajes:**
- "Estoy satisfecho con la vida en general (7/10)"
- "Estoy muy contento con mi familia (9/10)"
- "Quiero mejorar mi satisfaccion laboral"
- "Estoy insatisfecho con mis conexiones sociales (5/10)"

---

## Uso para Clasificacion Jerarquica

### Formato de Datos de Entrenamiento

Para entrenar el clasificador jerarquico BERT, crear datos de entrenamiento en este formato:

```csv
message,context,entity,sub_entity
"Mi objetivo es ganar 120k$ el proximo ano",professional,professional_aspirations,compensation_expectations
"Llevo 9 meses sobrio del alcohol",personal,health_and_wellbeing,addictions_or_recovery
"Quiero ser VP de Producto",professional,professional_aspirations,dream_roles
"Soy experto en estrategia de producto",learning,current_skills,skills
"Tengo sindrome del impostor moderado",psychological,confidence_and_self_perception,imposter_syndrome
"Estoy casado con un hijo pequeno",personal,personal_life,children
```

### Flujo de Clasificacion

1. **Nivel 1: Clasificacion de Contexto** (5 clases)
   - professional, learning, social, psychological, personal

2. **Nivel 2: Clasificacion de Entidad** (especifica por contexto)
   - Para professional: current_position, professional_experience, professional_aspirations, etc.
   - Para personal: personal_life, health_and_wellbeing, living_situation, etc.

3. **Nivel 3: Clasificacion de Sub-entidad** (especifica por entidad)
   - Para professional_aspirations: dream_roles, compensation_expectations, desired_work_environment, etc.
   - Para health_and_wellbeing: physical_health, mental_health, addictions_or_recovery

### Enrutamiento para Extraccion de Informacion

Una vez clasificado, enrutar al prompt de extraccion y endpoint API apropiado:

```
Mensaje: "Mi objetivo es ganar 120k$ el proximo ano"
↓
Contexto: professional
↓
Entidad: professional_aspirations
↓
Sub-entidad: compensation_expectations
↓
Extraer: {target_base_salary: 120000, currency: "USD", timeframe: "next year"}
↓
Almacenar: POST /api/harmonia/professional/professional-aspirations
```

---

## Notas

- Algunos mensajes pueden mapear a multiples rutas (ej: "Quiero ser VP y ganar 300k$" mapea tanto a dream_roles como a compensation_expectations)
- La estructura jerarquica permite degradacion elegante: si la clasificacion de sub-entidad es incierta, caer al nivel de entidad
- Los limites de contexto pueden solaparse (ej: "Quiero equilibrio vida-trabajo" podria ser psychological > values O personal > lifestyle_preferences)
- Siempre preferir el nivel de clasificacion mas especifico posible para mejor extraccion de informacion
