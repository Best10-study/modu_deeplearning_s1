# 5. Lec 05 Logistic (Regression) Classification

#### 2018.09.25(화)

## Intro

Logistic(Regression) Classification.

1. 분류 알고리즘 중에서 정확도가 매우 높은 알고리즘으로 알려져 있다.
2. 실무에서도 정말 많이 쓰이는 Classification model 이다.
3. 이 수업의 궁극적인 목표인 Neural Network와 Deep learning 의 매우 중요한 요소.



Endrew Eg 교수님 강의를 봅시다.

### Regression

- Hypothesis : $H(X) = XW$
- Cost : $cost(W) = \frac{1}{m}\sum(WX-y)$
- Gradient decent : $W := W-\alpha \frac{\alpha}{\alpha W}cost(W)$

우리는 이 세가지만 알면, Linear Regression 을 구현도 할 수 있고, 이해도 된다.근데, Classification도 이와 유사하다.

### Classification

-> Regression은 숫자를 예측하는 것이라면, Binary Classification은 둘 중 하나의 정해진 category를 정해주는 것.

- 활용 -> 0,1 encoding
  - Spam Detection : Spam(1) or Ham(0)
  - Facebook feed: show(1) or hide(0)
  - Credit Card Fraudulent Transaction detection: legitimate(0) or fraud(1)

### Pass(1) / Fail(0) based on study hours

-> 1, 0으로 나누는 classification의 scatter를 보면 왠지 Linear Regression 으로도 충분히 할 수 있을 것처럼 보인다.

Linear Regression의 결과값을 Threadholds로 나누면 되지 않을까? 라는 생각.

실제로, 우리는 이런식으로 처리할 것이지만, 이것에 조금 문제가 있다. 

- 문제점 예시
  1. 만약, input data 중 극명한 값을 가진 data가 있다면,model의 결과가 다소 달라질 수 있다. (기울기가 달라지기 때문에 -> 4시간만 공부해도 합격인데, 누군가 50시간을 공부하고 합격하고, 이를 가지고 Linear Regression 모델을 만들게 된다면, 50시간을 공부해도 결과는 pass(1)이기 때문에, model 의 기울기가 많이 낮아질 것 -> 합격할 사람도 떨어지는 결과가 나옴)
  2. We know Y is 0 or 1 $H(x) = Wx + b$ 여기서 H(x)가 0보다 훨신 작거나, 1보다 훨씬 큰 값이 결과로 나올 수 있다. ex) $H(x) = 0.5x + 0$ 에서, x가 100일 경우, H(x)는 50이 되어버리고, 그러면 우리가 분류하려고 했던 0,1 classification에서는 뭔가 문제가 있어 보인다. 

그래서, 우리는 이러한 문제점을 해결해주기 위해 그 결과값을 0~1 사이의 값으로 만들고 싶다.

그리하여, $ Z = Wx + b$ 로두고, 과연 g(Z) 를 0~1사이로 만드는 함수 g(Z)를 열심히 찾기 시작!

$g(x) = \frac{1}{(1+e^{-z})}$ 라는 함수를 발견하였다.이를 , Sigmoid라고 부른다. -> 결과값이 0~1 사이인 S 형의 그래프( Logistic function 이라고도 부른다.)

### Logistic Hypothesis

$ H(X) = \frac{1}{1+e^{-W^{T}X}}$

로 만들 수 있다.!

우리는, 이를 이용하는데, 그 다음 단계는 바로, Cost 를 구하고, 이를 최소화하는 방법 을 구해야 겠죠. 

## Cost function & gradient decent

### cost

$$ cost(W,b) = \frac{1}{m}\displaystyle\sum^m_{i=1}(H(x^{(i)})-y^{(i)})^2 \ when\ H(x)=Wx+b$$

이의 장점? -> 어느 점에서 시작한다고 하더라도 Minimum을 찾을 수 있다.

그런데, 우리의 가설이 조금 바뀌었다. $ H(X) = \frac{1}{1+e^{-W^{T}X}}$  

이 $H(x)$를 넣고 cost function을 넣으면, 울퉁불퉁한 function이 그려진다! ( 어떻게 그려볼지 상상 !) 

--> 이 경우, 학습의 초기값이 어디냐에 따라서 최소값이 달라질 수 있다.(문제점) : 이런 지점을 Local minimum 이라고 한다. 하지만, 우리가 알아야 하는 값은 Global minimum!

__우리는 Cost function에서  언제나 Global minimum 을 얻을 수 있도록 바꾸어줘야 한다.__

### New cost function for logistic

$cost(W) = \frac{1}{m}\sum c(H(x),y)$

$c(H(x),y) = \begin{cases} -log(H(x)) & : y=1 \\ -log(1-H(x)) & : y = 0 \end{cases} $

$ C(H(x),y) = - ylog(H(x)) - (1-y)log(1-H(x))$ # if condition 을 없애기 위함. 

>  y가 1일때와 y가 0일때로 나누어서 cost function을 새로 정의하자. ->  H(x)에는 e^ 가 들어가게 되는데, 이 부분 떄문에 local minimum 이 발생한다. 그래서, 이걸 평평하게 펴기 위해 log를 씌워준다.
>
> Why?? 왜 y=1 일때와 y=0 일때로 나누어서 처리를 하는가???
>
>  함수를 직접 그려보면 이해가 빠름 ->
>
>  y=1일때 H(x) = 1이라면, cost(1) = 0, H(x) = 0 -> cost = $\infin$ 가 된다.  
>
>  y=0일때 H(x) = 0이라면, cost(1) = 0, H(x) = 0 -> cost = $\infin$ 가 된다.
>
> 이 그래프를 합쳐주면 Convex function 이 그려짐 -> minimize 가능

### Minimize cost  - Gradient decent algorithm

$ C(H(x),y) = - ylog(H(x)) - (1-y)log(1-H(x))$

우리가 minimize를 하기 위해서는, 각 점에서의 기울기를 파악해야 한다.

$ W := W -\alpha \frac{\alpha}{\alpha W}cost(W)$

~~~python
# cost function
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis)))

# Minimize
a = tf.Variable(0.1) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
~~~

