# 4. Linear Regression cost 함수 최소화

#### 2018.09.18(화)

우리의 과제는 W,b를 구해서 Modeling을 한 후, 거기다가 X라는 테스트를 넣어서 Y를 예측하는건데, 그때, 오류를 최소화하는 것이 목표. 즉, Cost Function 을 어떻게 최소화 하느냐가 중요한 문제이다.

이번 강의에서는 Hypothesis  function 을 간략하게 $ H(x) = Wx$ 로 정의를 하고 cost function 또한 $cost(W) = \frac{1}{m}\sum_{i=1}^{m}(Wx^{(i)}-y^{(i)})^2$  로 정의한다 !

W가 0일때 cost를 계산하면, 0이 나온다.와 W가 1일때로 계산하면 4.67, W가 2일때로 계산하면 4.67

즉, CostFunction을 계산하면 2차함수정도가 나온다.

이는  Convex 함수임을 알 수 있음. 그러면 최소값이 존재한다는 것을 알 수 있다!!

### Gradient Descent Algorithm

$\alpha = learning\ rate$ 

$W := W - \alpha\frac{\delta}{\delta W}cost(W)$   

W에 대한 미분을 해결하여 주면

$W := W - \alpha\frac{1}{m}\sum^m_{i=1}(Wx^{(i)}-y^{(i)})x^{(i)}$ 의 꼴로 나오게 된다. 

이를 계속 실행시켜주면 W가 계속해서 변화하게 됨 그것이 학! 습!

-> Cost function 을 미분해서 양수이면 음수의 방향으로 쪼금, 음수이면 양수의 방향으로 쪼끔! 씩 이동

이때 쪼끔! => $\alpha$  로 정의되는것

즉 이런식으로 학습하므로써, 우리는 모델을 만들수 있는것!

하지만, 우리는 조금 생각해 볼 필요가 있다.

### convex function

3차원의 function 에서, 우리는 Gradient Descent Algorithm 이 잘 작동하는지 잘 생각해 보아야 한다.

우리가 만든 cost function은 3차원에서도 convex function 이기 때문에, 어느 점에서 시작하더라도 최저값을 찾아낼 수 있다.

우리는 우리가 만든 cost function 이 convex function 인지만 확인해 준다면 된다!

convex function이 아니라면,  초기값이 무엇이냐에 따라서 답이 달라질 수 있기때문에!

