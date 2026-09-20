# Targeted prior-art audit: obstacle problems, TT compression, certified feasibility

Проверено 20 сентября 2026. Это ограниченный аудит ближайших работ, а не доказательство отсутствия аналогов. Страницы ниже — страницы указанного открытого PDF, начиная с 1; номера разделов HTML относятся к явно указанной версии. Издательскую версию и препринт не считаем автоматически тождественными.

## 1. Три обязательных ближайших источника

### G — Garreis–Ulbrich: low-rank obstacle algorithms уже существуют

**Sebastian Garreis, Michael Ulbrich.** *Constrained Optimization with Low-Rank Tensors and Applications to Parametric Problems with PDEs*. SIAM Journal on Scientific Computing 39(1), A25–A54 (2017). [DOI / publisher](https://doi.org/10.1137/16M1057607), [проверенный полный препринт PDF](https://optimization-online.org/wp-content/uploads/2016/01/5301.pdf).

- **Проверено в тексте:** §3.2 — параметрическая obstacle VI и дискретная квадратичная задача с покомпонентными ограничениями; §4.2, pp.20–23 — primal interior-point algorithm, HT/TT-арифметика, AMEn и Newton–Schulz. Это непосредственный antecedent связки obstacle + Galerkin + low rank.
- **Feasibility:** p.22 задаёт допустимую длину шага через максимум отношения направления к slack. Реализация ищет максимум методом MCS, который сами авторы называют “non-rigorous” (два слова цитаты). Поэтому нельзя приписывать реализации сертификацию всех ограничений после сжатия.
- **Convergence:** конец §4.2, p.23, и заключение p.29 прямо оставляют строгую теорию inexact алгоритмов с truncation/error control для последующей работы. Упомянутые локальные условия для Newton–Schulz не равны такой теореме.
- **Rank/mesh:** ранги контролируются усечением; §5.2 сообщает численные настройки и результаты. Проверенный текст не даёт a priori ранговой оценки obstacle solution или mesh-uniform оценки полного алгоритма.
- **Граница проверки:** финальная журнальная библиография подтверждена; полный издательский PDF оказался недоступен. Конкретные утверждения о пропусках относятся к открытому препринту.

### A — Antil–Dolgov–Onwunta: regularization convergence нужно отделять от TT iteration convergence

**Harbir Antil, Sergey Dolgov, Akwum Onwunta.** *Smoothed Moreau-Yosida Tensor-Train Approximation of State-Constrained Optimization Problems Under Uncertainty*. Numerical Linear Algebra with Applications 32(4) (2025). [Полный издательский HTML / DOI](https://doi.org/10.1002/nla.70028), [arXiv](https://arxiv.org/abs/2301.08684), [проверенный PDF](https://arxiv.org/pdf/2301.08684).

- **Проверено в финальном тексте:** §3, Algorithm 1 — inexact projected Newton, smoothed Moreau–Yosida penalty и TT-cross. Одна `tol` используется для stopping и TT approximation/truncated SVD.
- **Теорема:** Appendix A, Theorem 1; в PDF-препринте это §3.1, Theorem 3.4, pp.5–10. Для linear-quadratic задачи, Slater и дополнительной регулярности получена strong convergence решений регуляризованных задач при согласованном пределе $\gamma\to\infty$, $\varepsilon_\gamma=o(\gamma^{-1/2})$. Это не теорема о конечных TT-итератах с независимыми ошибками cross, truncation и неточного Newton.
- **Feasibility:** конечный penalty допускает нарушения. §4.2 явно сообщает вероятность нарушения 0.0139223. §4.3 заменяет VI penalized PDE; не следует переносить на этот нелинейный пример linear-quadratic theorem.
- **Rank/mesh:** §4 и Table 1 дают эмпирические ранги/ошибки; Appendix C обсуждает рост рангов при уменьшении smoothing. Проверенных uniform-in-mesh или obstacle-specific rank bounds не найдено в этих разделах.

### K — Keith–Surowiec: pointwise-feasible reconstruction и теория LVPP уже есть

**Brendan Keith, Thomas M. Surowiec.** *Proximal Galerkin: A structure-preserving finite element method for pointwise bound constraints*. Foundations of Computational Mathematics, online 2024. [arXiv / bibliographic record](https://arxiv.org/abs/2307.12444), [проверенный полный HTML v7](https://arxiv.org/html/2307.12444v7).

- **Теория:** §4.5, Theorem 17, (70)–(72): energy descent и $H^1$-convergence точных continuous LVPP subproblems при $\sum_k\alpha_k=\infty$; оценка squared energy error через начальную Bregman distance и $\sum\alpha_k$.
- **Feasibility:** §4.6, Algorithm 3/(75) различает primal $u_h$ и reconstruction $\widetilde u_h=\phi+\exp\psi_h>\phi$. Последняя pointwise-feasible. Primal variable в общем сохраняет лишь cell averages; §5.3/Remark 31 при подходящей mass lumping даёт nodal bounds. Для P1/Q1 nodal bounds переходят в pointwise bounds. Эти гарантии нельзя смешивать.
- **Mesh:** §4.8 показывает устойчивые iteration counts; Appendix B даёт анализ linearized equations и uniform inf–sup. Его вступление явно откладывает полный a priori анализ nonlinear subproblems и полное доказательство mesh-independence метода.
- **Overlap limit:** TT-округление не является частью указанного алгоритма; сходимость exact LVPP не покрывает автоматически возмущение экспоненты сжатием. Boundary assumptions теоремы и нулевой Dirichlet trace требуют отдельного сопоставления.

## 2. Ближайшие проверки positivity / certified error / rank

### B1 — Предшественник простого rank-one сдвига

**Stanislav Budzinskiy.** *Quasioptimal alternating projections and their use in low-rank approximation of matrices and tensors*. Numerische Mathematik 157, 1491–1535 (2025). [Издательская версия](https://doi.org/10.1007/s00211-025-01483-6), [arXiv:2308.16097, проверенный полный текст](https://arxiv.org/html/2308.16097). Номера разделов и теорем ниже относятся к arXiv-версии.

§5, Theorems 5.3, 5.6–5.7 доказывают локальную сходимость quasioptimal alternating projections при геометрических условиях, включая transversality/super-regularity и ограничения на quasioptimality constants. §6.4.1 уже предлагает scalar shift-projection: прибавление скаляра к элементам повышает matrix rank максимум на один. Величина сдвига зависит от отрицательного минимума; обсуждаются вычисляемые по выборке суррогаты. §6.4.2 отдельно рассматривает nonnegative TT/QTT approximation. Следовательно, ни «сдвиг на константу», ни «nonnegative TT», ни «convergence with quasioptimal projections» сами по себе не годятся как заявка новизны. Теоремы не являются global obstacle-optimization guarantees и не дают mesh-independent констант для нашего contact set. Проверены разделы, а не только abstract.

### B2 — Entrywise TT approximation имеет свою теорию

**Stanislav Budzinskiy.** *Entrywise Tensor-Train Approximation of Large Tensors via Random Embeddings*. SIAM Journal on Matrix Analysis and Applications 46(2), 984–1005 (2025). [DOI](https://doi.org/10.1137/24M1649186), [полный текст v3](https://arxiv.org/html/2403.11768).

§1.3, Theorem 6 даёт существование TT approximation с max-norm error через factorization quasinorm и рангом порядка $c_d\varepsilon^{-2}\log(2e\prod n_j)$. Автор предупреждает о плохой размерностной зависимости доступной оценки $c_d$; §3 рассматривает coherence. Это важный antecedent любой фразы «первый entrywise TT certificate». Однако existence/a priori bound для подходящей аппроксимации отличается от детерминированного a posteriori envelope для фактически выполненного rounding. В проверенных разделах нет obstacle feasibility или boundary-preserving energy repair.

### P — Современный mesh/p-robust obstacle baseline

**Ioannis P. A. Papadopoulos.** *Hierarchical proximal Galerkin: a fast hp-FEM solver for variational problems with pointwise inequality constraints*. [arXiv:2412.13733, полный текст v4, 6 August 2026](https://arxiv.org/html/2412.13733).

Проверены §1.2, §§4.1–4.6 и примеры §6. Иерархический p-FEM, tensor-product domains, sparse Newton systems и block preconditioning; численно демонстрируются hp-robust outer iterations и слабый рост Krylov counts. Здесь tensor-product structure не означает TT-compressed solution. Этот источник обязателен как современный baseline для заявления об эффективности obstacle solver; нельзя сравниваться только с плохо обусловленным непрекондиционированным методом.

### S — Nonnegative TT prior art, проверка пока abstract-only

**E. M. Shcherbakova, E. E. Tyrtyshnikov.** *Fast Nonnegative Tensor Factorizations with Tensor Train Model*. Lobachevskii Journal of Mathematics 43, 882–894 (2022). [Издательская страница / DOI](https://doi.org/10.1134/S1995080222070228).

Официально проверены библиография и abstract о nonnegative TT/CP/Tucker factorization. Полный первичный текст не проверен; утверждения о его convergence, stopping certification и rank bounds здесь намеренно не делаются. B1 ссылается на него в §6.4. Этот пункт остаётся конкретным обязательством дальнейшего чтения, а не основанием заявлять отсутствие сертификатов.

### Q — Модули SVD-факторов уже использовались в NMF

**Hanli Qiao.** *New SVD based initialization strategy for Non-negative Matrix Factorization* (2014). [Полный первичный PDF, arXiv:1410.2786](https://arxiv.org/pdf/1410.2786).

Проверен §2.2, p.4: инициализация SVD-NMF использует $W_0=|U|$, $H_0=|\Sigma V^T|$. Поэтому конструкция $|U|\Sigma|V|^T$, совпадающая с суммой модулей SVD rank-one terms, имеет явный предшественник. Из неравенства треугольника самостоятельно следует её envelope-свойство для реконструированной матрицы; не следует приписывать Qiao доказательство obstacle/energy theorem. Наш возможный вклад может касаться оценки энергии оболочки, сохранения границы или спектральной группировки, но не самой идеи взятия модулей SVD-факторов. Полная новизна этих дальнейших оценок пока не установлена.

## 3. Матрица overlap

`не установлено` означает отсутствие проверенного результата в рассмотренных разделах, а не утверждение об отсутствии результата во всей литературе.

| Источник | Obstacle/VI | TT/HT compression | Feasibility в конечном вычислении | Что именно сходится | Mesh/rank guarantee для compressed obstacle |
|---|---|---|---|---|---|
| G | Да | Да | Поиск допустимого шага без rigorous global certificate | Полная truncation-теория отложена в препринте | Не установлено |
| A | State constraints; VI example | TT-cross | Penalty violations допустимы | Regularization limit | Не установлено |
| K | Да | Нет | Reconstruction; primal требует оговорок | Exact continuous LVPP | Не TT-результат |
| B1 | Общие constraints | TT examples | Зависит от projections/stopping | Local QAP | Не установлено |
| B2 | Нет | Да | Max-norm approximation theory | Не obstacle iteration | A priori rank, без obstacle |
| P | Да | Tensor-product FEM | PG-постановка | Практический hpG | Нет TT-анализа |
| S | Нет в abstract | Да | Abstract-only | Не проверено | Не проверено |
| Q | Нет | SVD/NMF | Absolute-factor initialization | Не obstacle iteration | Не установлено |

## 4. Условия, которые должна закрыть наша теория

Следующие пункты — требования к проекту и собственные выводы аудита, а не уже доказанные результаты.

1. **Уровень feasibility.** Указать конечную SPD QP $\min_{u\ge\psi} \frac12u^TAu-f^Tu$, множество ограниченных координат и правило переноса к функции. Для P1/Q1 неотрицательной partition-of-unity basis nodal inequality относительно интерполированного obstacle переносится внутрь элементов; для произвольного high-order Galerkin — нет. Непрерывный obstacle и его интерполяция требуют отдельной consistency оценки.

2. **Сертификат фактической компрессии.** Для данного rounded tensor $\widehat u$ получить доказанную покомпонентную мажоранту $|u-\widehat u|\le E$ без обхода полного массива. Указать, используются ли Frobenius remainder, corewise telescoping или weighted norm; разделить истинный upper bound и эвристическую оценку по samples. Floating-point certificate требует directed rounding/интервальных или явно учтённых roundoff bounds; exact-arithmetic lemma не сертифицирует программу автоматически.

3. **Граница и цена repair.** Простое прибавление $\delta\mathbf1$ портит Dirichlet trace, если он включён в массив. Bubble $b_h\ge0$, $b_h|_{\partial\Omega}=0$, допускает $\widehat u+\delta b_h$, только если предварительно доказано $|u-\widehat u|\le\delta b_h$. Малость обычного Frobenius error этого не обеспечивает у границы. Нужна оценка полной поправки в energy norm, включая энергетику самой ошибки округления.

4. **Полная возмущённая итерация.** Теорема должна включать ошибки operator action, inner solve, nonlinear/projection approximation, truncation и feasibility repair. Проверить достаточные summable/relative tolerances и вычислимость stopping rule. Generic inexact contraction theorem является опорной леммой; substantive result — проверка её условий для сжатого obstacle solver.

5. **Mesh robustness отделить от fixed-grid convergence.** Все константы показывать с зависимостью от $h,d$, condition number и весов. Frobenius-to-energy norm equivalence на каждой сетке сама по себе не даёт mesh robustness. Uniform contraction должен опираться на явно заданный preconditioner/metric; экспериментальное плато iteration counts не заменяет такого условия.

6. **Rank accounting отделить от compressibility.** Для repair допустима алгебраическая оценка роста TT ranks через ранг envelope/bubble. Она не доказывает низкие ранги решения, bounded ranks всей траектории или отсутствие curse of dimensionality. Предположения о compressibility свободной границы/данных должны быть самостоятельными и проверяемыми. Не сжимать уже repaired результат без нового сертификата.

7. **Primal–dual stopping.** Если используется gap/residual certificate, проверять dual feasibility и знак multiplier, точность применения $A^{-1}$ и error budget для скалярных contractions. Feasible primal alone не является доказательством близости к оптимуму; малый computed residual без error accounting тоже.

## 5. Безопасная формулировка исследовательской ниши

Предварительная цель: **детерминированное подтверждение допустимости после TT-компрессии для дискретной obstacle QP, с поправкой, совместимой с граничными условиями, и явной оценкой её вклада в energy error и сходимость inexact iterations**. Это сочетание требований не было установлено в проверенных результатах. Предложение остаётся условным, пока не завершены проверка S и сопоставление собственных лемм с existing max-norm approximation, QAP и generic inexact optimization.

Не заявлять «первое решение obstacle через TT», «первая positivity-preserving low-rank approximation», «первый rank-one repair», «mesh-independent TT solver» или «разрушение проклятия размерности» только из наличия компактной записи операций.
