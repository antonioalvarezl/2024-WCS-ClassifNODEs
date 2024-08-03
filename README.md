# 2024-WCS-ClassifNODEs
Codes from the article "Controlled cluster-based classification with neural ODEs", by Antonio Álvarez-López, Rafael Orive-Illera, and Enrique Zuazua

\textbf{Objective}: 
For $d$ and $N$ given, find the complexity (number of discontinuities $L$) required by the control functions $(w,a,b)$ of a neural ODE 
$$
\dot x = w(t)\sigma(a(t)\cdot x+b(t)),\qquad t\in(0,T)
$$
to classify a dataset $\{(x_i,y_i)\}_{i=1}^N$, where $x_i\sim U([0,1]^d)$ and $y_i\in\{0,1\}$ are randomly chosen for all $i$. 

Classification is understood as finding some controls $(w,a,b)$ such that the flow map $\Phi_T$ of the neural ODE satisfies 
$$
\Phi_T(x_i;w,a,b)^{(d)}>1\quad\text{for all }x_i\text{ such that }y_i=1,
$$
and 
$$
\Phi_T(x_i;w,a,b)^{(d)}<1\quad\text{for all }x_i\text{ such that }y_i=0.
$$
Additionally, we study the dependence of $L$ with $d$ and $N$.
