# 2024-WCS-ClassifNODEs

Codes from the article "Controlled cluster-based classification with neural ODEs," by Antonio Álvarez-López, Rafael Orive-Illera, and Enrique Zuazua.

**Objective:**

For given $d$ and $N$, find the complexity (number of discontinuities $L$) required by the control functions $(w,a,b)$ of the neural ODE

$$
\dot{x} = w(t)\sigma(a(t) \cdot x + b(t)), \quad t \in (0, T)
$$

to classify a dataset $\{(x_i,y_i)\}_{i=1}^N$, where $x_i \sim U([0,1]^d)$ and $y_i \in \{0,1\}$ are randomly chosen for all $i$.

Classification is understood as finding some controls $(w,a,b)$ such that the flow map $\Phi_T$ of the neural ODE for some fixed $T>0$ satisfies:

$$
\Phi_T(x_i;w,a,b)^{(d)} > 1 \quad \text{for all } x_i \text{ such that } y_i = 1,
$$

and 

$$
\Phi_T(x_i;w,a,b)^{(d)} < 1 \quad \text{for all } x_i \text{ such that } y_i = 0.
$$

Additionally, we study the dependence of $L$ on $d$ and $N$.
