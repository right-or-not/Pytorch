# Sine and Cosine Positional Encoding

> From article *Attention is All You Need*.

## Encoding Formulas

- $PE(pos, 2i) = \sin(\frac{pos}{10000^{\frac{2i}{dim}}}})$
- $PE(pos, 2i+1) = \cos(\frac{pos}{10000^{\frac{2i}{dim}}}})$

## Understanding of Encoding Formulas

### Use Sine and Cosine to Encoding

For the same value of `i`, $PE(pos, 2i)$ and $PE(pos, 2i+1)$ have the same **Angular Frequency** $\omega = \frac{1}{10000^{\frac{2i}{dim}}}$. When the value of `i` is selected, $PE(pos, 2i)$ and $PE(pos, 2i+1)$ only depend on `pos`.

Let:

$$
\omega = \frac{1}{10000^{\frac{2i}{dim}}}
$$

and 

$$
PM(pos)_i
= 
\begin{pmatrix}
PE(pos, 2i) \\
PE(pos, 2i+1)
\end{pmatrix}
=
\begin{pmatrix}
\sin(pos \cdot \omega) \\
\cos(pos \cdot \omega)
\end{pmatrix}
$$

What we want is **Shifting Position** via **Matrix Multiplication**, which means:

$$
PM(pos + k)_i = M(k) \cdot PM(pos)_i
$$

Likewase,

$$
\begin{pmatrix}
\sin((pos + k) \cdot \omega) \\
\cos((pos + k) \cdot \omega)
\end{pmatrix}

=

\begin{pmatrix}
&\cos(k \cdot \omega) &-\sin(k \cdot \omega) \\
&\sin(k \cdot \omega) &\cos(k \cdot \omega) \\
\end{pmatrix}

\cdot

\begin{pmatrix}
\sin(pos \cdot \omega) \\
\cos(pos \cdot \omega)
\end{pmatrix}
$$

In Machine Learning, it is simple to calculate matrix multiplication, while sine and cosine make it possible to shift the position via matrix multiplilcation.


### Angular Frequency is A Function of `i`

Above formulas are all based on:

$$
\omega = \frac{1}{10000^{\frac{2i}{dim}}}
$$

But why we choose $\omega = f(i)$? With different $\omega$, the machine can understand the relationship between the words.

| Value of $\omega$ | Position Difference |                       Use                        |
| ----------------- | ------------------- | ------------------------------------------------ |
|       small       |        obvious      | understand the difference between word and word  |
|       large       |        smooth       | understand the Continuity between neighbor words |
