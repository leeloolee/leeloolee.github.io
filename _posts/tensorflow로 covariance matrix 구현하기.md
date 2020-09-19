# tensorflow로 covariance matrix 구현하기

## 파이썬으로 convariace 구하기

https://donghwa-kim.github.io/covariance.html 요거 베낀거

이 글은 파이썬을 활용한 Covariance를 구하는 과정을 설명한 글입니다. 

데이터의 variance는 크게 3가지 방식으로 구할수 있습니다. (full, diag, spheircal) 아래의 random으로 생성된 데이터를 예로 들어보겠습니다. 



```python
np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C), 
	.7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
X
```



```
array([[ 0.68026725, -0.01634235],
       [ 3.80951844,  0.79848348],
       [-1.6613724 , -0.57766695],
       ...,
       [-5.86152028,  3.06842556],
       [-5.01893361,  3.11090369],
       [-6.79933099,  2.08232074]])
```



Full Covariance

 * Covariance를 산출

   ```python
   # np.atleast_2ds : 1-dim array가 들어오면 2-dimensional로 바꿔주는 함수 
   # np.cov: covariance
   np.atleast_2d(np.cov(X.T))
   ```

   

   ```
   array([[10.51401566, -4.13489296],
          [-4.13489296,  2.57580056]])
   ```

   



Diagonal Variance

* 변수별로 variance 산출

  ```python
  # axis = 0 : 변수별로 variance를 산출
  # ddof : Degrees of Freedom
  np.var(X, axis = 0, ddof = 1)
  ```

  

Spherical Variance

* 변수별로 산출된 variance에 대한 평ㄱㄴ

  ```python
  np.var(X, axis = 0, ddof = 1).mean()
  ```

  ```
  6.544908108895438
  ```



이런

https://www.kdnuggets.com/2018/10/preprocessing-deep-learning-covariance-matrix-image-whitening.html



