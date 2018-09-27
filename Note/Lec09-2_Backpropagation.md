# 9.Lec09-2_Backpropagation

> Neural Network 1: XOR 문제와 학습방법, Backpropagation (1986 breakthrough) 

#### 2018.09.26(수)

### How can we learn W1, W2, B1, B2 from training data?

Gradient Descent Algorithm 을 가지고 학습시킬 수 있다!

cost함수가 convex 함수라면...!

그런데, 여기서 우리가 mimimum한 값을 찾기 위해서는 미분을 하여 기울기를 알아야 하는데, NN으로 가면서 미분값이 지나치게 복잡해진다.! -> 노드가 한개만 있는게 아니라, sigmoid도 들어가고, 여러개의 node가 서로 겹치고 하면서, 모든 node의 미분값을 알아야 최종 cost를 구할 수 있게 됨... -> 계산량이 너무 많아진다.

그리하여, minsky교수는 이것이 불가능하다고 말한것.

### Backpropagation(Chain rule)

하지만, 이것이 Backpropagation 에 의해 해결된다.

우리의 모델을 통해 도출된 error값을 가지고 다시 backward 방향으로 가며 학습

