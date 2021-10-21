"""
Autor: Juan Pablo Echeagaray González.
Fecha: 06/10/2021
No me considero autor intelectual del presente código, esta compilación
de funciones de la librería scipy no es más que el intento de un estudiante
de facilitarse un curso de estadística básica.
"""

import scipy.stats as stats
from math import factorial

# Teoría de conteo


def permutations(n: int, k: int) -> int:
    """
    ## Número de permutaciones de k elementos en n

    ### Args:
        - n (int): Número de elementos en total
        - k (int): Número a partir del cual hacer permutaciones

    ### Returns:
        - int: (n k)
    """
    perm = factorial(n) / factorial(n - k)
    return perm


# Distribuciones de variable discreta


def binom_pmf(x: int, n: int, p: float) -> float:
    """
    PMF de variable con distribución binomial

    Args:
        x (int): Número de éxitos esperados en la muestra
        n (int): Tamaño de la muestra
        p (float): Probabilidad de éxito

    Returns:
        float: P(X = x)
    """
    if n == 0:
        raise ValueError("n debe de ser mayor a 0")
    elif (p < 0) or (x < 0):
        raise ValueError("p y x deben de ser mayor o igual a 0")
    return stats.binom.pmf(x, n, p)


def binom_cdf(x: int, n: int, p: float) -> float:
    # Regresa la probabilidad P(X <= x)
    return stats.binom.cdf(x, n, p)


def multinom_pmf(X: list[int], P: list[float]) -> float:
    n = sum(X)
    prob = stats.multinomial.pmf(X, n, P)
    return prob


def geom_pmf(x: int, p: float) -> float:
    return stats.geom.pmf(x, p)


def geom_cdf(x: int, p: float) -> float:
    return stats.geom.cdf(x, p)


def hypergeom_pmf(x: int, N: int, n: int, k: int) -> float:
    """
    ## PMF de una variable con distribución hipergeométrica.

    ### Args:
        - x (int): Número de éxitos esperados en la muestra
        - N (int): Tamaño de la población
        - n (int): Tamaño de la muestra
        - k (int): Número de éxitos en la población

    ## Returns:
        - float: P(X = x)
    """
    return stats.hypergeom.pmf(x, N, n, k)


def hypergeom_cdf(x: int, N: int, n: int, k: int) -> float:
    """
    ## Cálculo de probabilidad acumulada de una variable hipergeométrica.

    ### Args:
        - x (int): Número de éxitos esperados en la muestra
        - N (int): Tamaño de la población
        - n (int): Tamaño de la muestra
        - k (int): Número de éxitos en la población

    ### Returns:
        - float: P(X <= x) (Acumulada)
    """
    prob = stats.hypergeom.cdf(x, N, n, k)
    return prob


def poisson_pmf(x: int, mu: float) -> float:
    """
    ## PMF de una variable Poisson.

    ### Args:
        x (int): # Número de sucesos a presenciar en tiempo t,
                debe de ser mayor a 0
        mu (float): Media de sucesos en un tiempo t

    ### Returns:
        float: P(X = x)
    """
    prob = stats.poisson.pmf(x, mu)
    return prob


def poisson_cdf(x: int, mu: float) -> float:
    """
    ## Probabilidad acumulada de variable Poisson.

    ### Args:
        - x (int): # Número de sucesos a presenciar en tiempo t
        - mu (float): Media de sucesos en un tiempo t
                    (misma unidad de tiempo que x)

    ### Returns:
        - float: P(X <= x) (Acumulada)
    """
    prob = stats.poisson.cdf(x, mu)
    return prob


def multinomial_pmf(x, n: int, p: float) -> float:
    return stats.multinomial.pmf(x, n, p)


# Distribuciones de variable continua


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    """
    ## PDF de RV normal.

    ### Args:
        - x (float): Valor esperado de la RV
        - mu (float): Media de la RV
        - sigma (float): Desviación estándar de la RV

    ### Returns:
        - float: P(X = x)
    """
    return stats.norm.pdf(x, mu, sigma)


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    """
    ## Probabilidad acumulada de RV normal.

    ### Args:
        - x (float): Valor de la variable aleatoria
        - mu (float): Media de la variable aleatoria
        - sigma (float): Desviación estándar de la variable aleatoria

    ### Returns:
        - float: P(X <= x) (Acumulada)
    """
    return stats.norm.cdf(x, mu, sigma)


def gamma_pdf(x: float, alpha: int, beta: float) -> float:
    return stats.gamma.pdf(x, alpha, scale=beta)


def gamma_cdf(x: float, alpha: int, beta: float) -> float:
    """
    ## Cálculo de probabilidad acumulada de distribución gamma.

    ### Args:
        - x (float): Valor esperado de la variable
        - alpha (float): Parámetro de forma
        - beta (float): Parámetro de escala

    ### Returns:
        - float: P(X<=x) (Acumulada)
    """
    res = stats.gamma.cdf(x, alpha, scale=beta)
    return res


def exponential_pdf(x: float, beta: float) -> float:
    prob = stats.expon.pdf(x, scale=beta)
    return prob


def exponential_cdf(x: float, beta: float) -> float:
    prob = stats.expon.cdf(x, scale=beta)
    return prob


def t_student_cdf(t: float, df: int) -> float:
    """
    ## Cálculo de probabilidad acumulada de una variable t-student.

    ### Args:
        - t (float): Estadístico t, se obtiene de la fórmula:
            t = sqrt(n) * (X - mu) / s, con
            - n = número de observaciones en la muestra
            - X = media de la muestra
            - mu = media poblacional
            - s = desviación estándar de la muestra
        - df (int): Grados de libertad (n - 1)
            Donde n es el número de observaciones en la muestra

    ### Returns:
        - float: P(T<=t) (Acumulada)
    """
    prob = stats.t.cdf(t, df)
    return prob


def chi_square_cdf(x, df) -> float:
    """
    Cálculo de probabilidad acumulada de una variable Chi cuadrada.

    Args:
        x (float): valor esperado de chi-cuadrado
        df (int): grados de libertad (n - 1)
        Donde n es el número de observaciones en la muestra

    Returns:
        float : P(X<=x) (Acumulada)
    """
    return stats.chi2.cdf(x, df)


def f_cdf(x: float, df_num: int, df_den: int) -> float:
    """
    Cálculo de probabilidad acumulada de una variable F.

    Args:
        x (float): valor esperado de F
        df_num (int): grados de libertad numerador - 1
        df_den (int): grados de libertad denominador - 1
        Donde n es el número de observaciones en la muestra

    Returns:
        float : P(X<=x) (Acumulada)
    """
    return stats.f.cdf(x, df_num, df_den)

# Pruebas de hipótesis


def t_student_ppf(alpha: float, df: int, part: str) -> float:
    possible = ['left', 'right', 'center']
    if part not in possible:
        raise ValueError("Parte no válida")
    elif part == possible[0]:
        t_score = - stats.t.ppf(1 - alpha, df)
    elif part == possible[1]:
        t_score = stats.t.ppf(1 - alpha, df)
    elif part == possible[2]:
        t_score = stats.t.ppf(alpha / 2, df)
    return t_score


def normal_ppf(alpha: float, part: str) -> float:
    possible = ['left', 'right', 'center']
    if part not in possible:
        raise ValueError("Parte no válida")
    elif part == possible[0]:
        z_score = - stats.norm.ppf(1 - alpha)
    elif part == possible[1]:
        z_score = stats.norm.ppf(1 - alpha)
    elif part == possible[2]:
        z_score = stats.norm.ppf(alpha / 2)
    return z_score


def chi_ppf(alpha: float, df: int) -> float:
    return stats.chi2.ppf(1 - alpha, df)


def f_ppf(alpha: float, df1: int, df2: int) -> float:
    return stats.f.ppf(1 - alpha, df1, df2)
