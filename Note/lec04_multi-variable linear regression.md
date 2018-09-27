# LEC 04 - multi-variable linear regression

#### 2018.09.22(토)

__Linear Regression__

다시 기억할 것

- Hypothesis -> $H(x) = Wx + b$
- Cost function -> $ cost(W,b) = \frac{1}{m}\sum_{i=1}^{m}(H(x^{(i)})-y^{(i)})^2 $
- Gradient descent Algorithm ->  $W := W - \alpha\frac{1}{m}\sum^m_{i=1}(Wx^{(i)}-y^{(i)})x^{(i)}$ 



__만약, 우리가 하나의 Input variable 이 아니라 여러개의 Input variable 을 다루게 된다면??__

- __Hypothesis__
  - $H(x_1,x_2,x_3) = w_1x_1 + w_2x_2+w_3x_3 + b$
- __Cost function__
  - $cost(W,b) = \frac{1}{m}\sum^m_{I=1}(H(x_1^{(i)},x_2^{(i)},x_3^{(i)})-y^{i})^2$



__더 많은 경우도 동일!!__

__$H(x_1,x_2,x_3,…,x_n) = w_1x_1 + w_2x_2+w_3x_3+...+w_nx_n + b$__

 

그런데, 이렇게 하다보니 불편한 감이 있다... -> n값이 길어지면 수식이 너무 길어질텐데??

=> matrix 로 처리! 우리는  matrix의 곱셈을 이용하면 되겠지!

!주의할 것 하나. matrix 를 이용할 경우, $\matrix{(x_1 & x_2 & x_3)} *  \begin{pmatrix} w_1 \\ w_2 \\ w_3 \end{pmatrix} = \matrix{(x_1w_1 + x_2w_2+x_3w_3)}$

이므로, $H(X) = XW$ 가 된다. 즉, matrix 로 표현할때는 WX 가 아닌, XW 가 됨에 주의 



우리는 데이터를 Instance 라고 부르자.

여기서! matrix 의 장점이 하나 나온다.!!

우리가 데이터를 하나만 넣을건 아니다!! 그래서, 여러개의 Instance를 가지고 학습시킨다고 하면, 이또한 Matrix 로 나타낼 수 있다는 장점이 있지요.

예를 들면,

| $x_1$ | $x_2$ | $x_3$ | $Y$  |
| ----- | ----- | ----- | ---- |
| 73    | 80    | 75    | 152  |
| 93    | 88    | 93    | 185  |
| 89    | 91    | 90    | 180  |
| 96    | 98    | 100   | 196  |
| 73    | 66    | 70    | 142  |

라는 데이터가 있다고 치면, 이에 대한 Hypothesis로 나타낼 떄

$\begin{pmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23}\\ x_{31} & x_{32} & x_{33} \\ x_{41} & x_{42} & x_{43} \\ x_{51} & x_{52} & x_{53} \\  \end{pmatrix} *  \begin{pmatrix} w_1 \\ w_2 \\ w_3 \end{pmatrix} = \begin{pmatrix}{x_{11}w_1 + x_{12}w_2+x_{13}w_3 \\ x_{21}w_1 + x_{22}w_2+x_{23}w_3 \\ x_{31}w_1 + x_{32}w_2+x_{33}w_3 \\ x_{41}w_1 + x_{42}w_2+x_{43}w_3 \\ x_{51}w_1 + x_{52}w_2+x_{53}w_3 }\end{pmatrix}$

로 나타낼 수 있지요.

이는 matrix의 dimension으로 봤을때 $[5,3] * [3,1] = [5,1]$인것.

 

즉, 인스턴스의 수대로 row를 넣어서 matrix를 만들 수 있다는것. 그럼에도, W는 변화가 없다.

이렇게 하면 여러개의 인스턴스를 각각 한번씩 계산할 필요 없이, 긴 매트릭스를 하나 넣어서 한번에 곱해서 원하는 값을 얻을 수 있다는 장점이 있다.



__이쯤에서, 우리가 수학시간에 열심히 SVD 이니 하는 그런 것들을 왜 배웠는지 나온다.__

우리는 학습을 위해서, matrix*vector를 곱하는 경우가 많아지는데, 그 vector를 diagonal matrix 로 바꾸어서 계산하는것!

이를 통해 주성분 분석도 할 수 있고... 그렇다. 선형대수 다시 공부할 필요가 있다!!



__상기의 H(X) 식에 대해서, 우리는 보통 X와 H(X) 의 size는 대강 알 수 있다.__

- 그럼, W의 사이즈도 파악가능. How?
  - X의 경우, [number of instance, number of Var]
  - H(X)의 경우, [number of instance, number of Y]

그러므로, W의 Size는 -> [#of Var, Y]

우리는 대게 Y(결과값)을 한개로 출력하겠지 -> Y=1



__이를 일반화한다면?__ ( n개의 데이터를 써서, (#of Y)개의 결과값을 낸다면! )

$[n, \#\ of\ var] * [\#\ of\ var, \#\ of \ Y] = [n, \#\ of \ Y]$  가 되는것.

 

### $WX\ vs\ XW$

-  Lecture ( theory ):
  - $$ H(x) = Wx + b$$
- Implementation ( TensorFlow ):
  - $H(X) = XW $