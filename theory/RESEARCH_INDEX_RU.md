# Теория допустимого TT-сжатия для obstacle-задач

Актуальное продолжение: [SOLUTION_APPROXIMATION_RU.md](SOLUTION_APPROXIMATION_RU.md) — аппроксимация самого obstacle-решения по структуре данных: взаимодействия и спектральные моды, допустимое исключение слабых координат, проверяемая геометрия контакта, контрпример к безусловной относительной оценке.

Предыдущий этап: [INTERACTION_THEORY_RU.md](INTERACTION_THEORY_RU.md) — теорема для ограниченного порядка взаимодействий в согласованном Q1-Галеркине, связь с фактическим TT-округлением, обмен между рангом поправки и энергетической ценой, условная сходимость в фиксированном пространстве взаимодействий.

Предыдущая многомерная записка: [MULTIDIM_THEORY_RU.md](MULTIDIM_THEORY_RU.md) — квадратичная TT-поправка, аддитивный класс и контрпример к универсальной положительной поправке.

Исходная постановка, двумерные результаты и сертификаты решения: [RESEARCH_NOTE_RU.md](RESEARCH_NOTE_RU.md).

Материалы:

| Файл | Содержание |
|---|---|
| [SOLUTION_APPROXIMATION_RU.md](SOLUTION_APPROXIMATION_RU.md) | От структуры данных к аппроксимации решения; доказательства и ограничения |
| [solution_approximation_research.md](solution_approximation_research.md) | Сохранение контакта, явные условия на данные и ошибка ограниченного минимума |
| [solution_obstructions_research.md](solution_obstructions_research.md) | Низкопорядковая нагрузка и высокопорядковое решение ранга один |
| [approximation_related_work.md](approximation_related_work.md) | Проверка первичных источников по аппроксимации obstacle-задач |
| [SOLUTION_APPROXIMATION_REVIEW.md](SOLUTION_APPROXIMATION_REVIEW.md) | Внутренний аудит аппроксимационных теорем |
| [verify_solution_structure.py](../research/audits/verify_solution_structure.py) | Проверки хвостов, положительных проекторов и контрпримера |
| [solution_structure_results.json](../research/audits/solution_structure_results.json) | Результаты структурных проверок |
| [verify_contact_approximation.py](../research/audits/verify_contact_approximation.py) | Проверки контакта и ошибки ограниченного минимума |
| [contact_approximation_results.json](../research/audits/contact_approximation_results.json) | Результаты проверок контактной геометрии |
| [INTERACTION_THEORY_RU.md](INTERACTION_THEORY_RU.md) | Порядок взаимодействий вместо размерности; теорема о фактическом сжатии |
| [rounding_structure.md](rounding_structure.md) | Ортогональные проекторы, сравнение масс, измеримые структурная часть и хвост |
| [rounding_geometry.md](rounding_geometry.md) | Настоящие ошибки TT-SVD: парные взаимодействия и точный контрпример |
| [rounding_limits_literature.md](rounding_limits_literature.md) | Почему малой энергии хвоста недостаточно; первичные источники |
| [INTERACTION_REVIEW.md](INTERACTION_REVIEW.md) | Внутренний аудит нового этапа |
| [verify_interaction.py](../research/audits/verify_interaction.py) | Проверки взаимодействий и реального TT-SVD |
| [interaction_verification_results.json](../research/audits/interaction_verification_results.json) | Результаты этих проверок |
| [MULTIDIM_THEORY_RU.md](MULTIDIM_THEORY_RU.md) | Многомерное продолжение с основными доказательствами |
| [multidim_core_research.md](multidim_core_research.md) | Аддитивный класс, точные тождества в Q1-энергии, поправка ранга 3 |
| [multidim_polynomial_research.md](multidim_polynomial_research.md) | Квадратичная поправка, полиномы Бернштейна и ограниченный порядок взаимодействий |
| [multidim_obstructions.md](multidim_obstructions.md) | Строгие ограничения универсальных положительных поправок |
| [MULTIDIM_REVIEW.md](MULTIDIM_REVIEW.md) | Внутренний аудит многомерных доказательств |
| [verify_multidim.py](../research/audits/verify_multidim.py) | Проверки новых тождеств и конструкций, включая 64 координаты |
| [multidim_verification_results.json](../research/audits/multidim_verification_results.json) | Результаты многомерных проверок |
| [RESEARCH_NOTE_RU.md](RESEARCH_NOTE_RU.md) | Основная связная записка на русском |
| [spectral_envelope.md](spectral_envelope.md) | Спектральные группы: допустимость, ранг и энергия; перенос на Q1-Галеркин |
| [envelopes_rank.md](envelopes_rank.md) | TT-оболочки, граничная поправка, контрпримеры, многомерная SVD-конструкция |
| [certificates_convergence.md](certificates_convergence.md) | Сертификаты primal–dual ошибки и неточная проксимальная сходимость |
| [related_work.md](related_work.md) | Ограниченный аудит ближайшей литературы с указанием проверенных разделов |
| [verify_theory.py](../research/audits/verify_theory.py) | Малые численные проверки сертификатов, TT-оболочек, контрпримера и сходимости |
| [verification_results.json](../research/audits/verification_results.json) | Сохранённые результаты этих проверок |
| [verify_spectral.py](../research/audits/verify_spectral.py) | Отдельная проверка спектральной конструкции и Q1-переноса |
| [spectral_verification_results.json](../research/audits/spectral_verification_results.json) | Результаты спектральных проверок |
| [REVIEW.md](REVIEW.md) | Внутренний независимый аудит доказательств и исправлений |

Для запуска нужен Python 3 и NumPy. В текущей среде:

```bash
cd research/audits
python verify_theory.py
python verify_spectral.py
python verify_multidim.py
python verify_interaction.py
python verify_solution_structure.py
python verify_contact_approximation.py
```

Численные проверки используют плотные эталоны на малых массивах. Это проверка согласованности формул, не производительный PDE-решатель. Теоремы относятся к точной арифметике; программа не реализует интервальные сертификаты.

Научная новизна не подтверждена. Стандартные оптимизационные аргументы и известные конструкции явно отделены от потенциального вклада. Не доказаны малые ранги на всём классе PDE, эффективность внутренних TT-итераций и устранение зависимости стоимости от размерности.
