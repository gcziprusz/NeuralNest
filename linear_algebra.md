# Multiply Vector with Matrix 

# Option 1 (dot product)
#### Dot Product Calculation

Matrix \( A \) and vector \( \mathbf{x} \):

$$
A = \begin{bmatrix} 
2 & 3 \\ 
1 & 4 \\ 
0 & -1 
\end{bmatrix}, \quad 
\mathbf{x} = \begin{bmatrix} 5 \\ 2 \end{bmatrix}
$$

Dot product computation:

$$
\mathbf{y} = A \cdot \mathbf{x} = \begin{bmatrix} 
2 & 3 \\ 
1 & 4 \\ 
0 & -1 
\end{bmatrix} 
\cdot 
\begin{bmatrix} 
5 \\ 
2 
\end{bmatrix}
=
\begin{bmatrix} 
2 \cdot 5 + 3 \cdot 2 \\ 
1 \cdot 5 + 4 \cdot 2 \\ 
0 \cdot 5 + (-1) \cdot 2 
\end{bmatrix}
=
\begin{bmatrix} 
16 \\ 
13 \\ 
-2 
\end{bmatrix}
$$

---

# Option 2 (linear combinations)
#### Linear Combination Calculation

Matrix \( A \) and scalar weights \( w_1 = 5 \), \( w_2 = 2 \):

$$
A = \begin{bmatrix} 
2 & 3 \\ 
1 & 4 \\ 
0 & -1 
\end{bmatrix}, \quad 
\mathbf{w} = \begin{bmatrix} 5 \\ 2 \end{bmatrix}
$$

Linear combination:

$$
A \cdot \mathbf{w} = w_1 \cdot \text{col}_1 + w_2 \cdot \text{col}_2
$$

Breaking it down:

$$
A \cdot \mathbf{w} = 5 \cdot \begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix} + 2 \cdot \begin{bmatrix} 3 \\ 4 \\ -1 \end{bmatrix} = \begin{bmatrix} 10 \\ 5 \\ 0 \end{bmatrix} + \begin{bmatrix} 6 \\ 8 \\ -2 \end{bmatrix} = \begin{bmatrix} 16 \\ 13 \\ -2 \end{bmatrix}
$$



