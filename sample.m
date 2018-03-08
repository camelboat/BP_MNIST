images = loadMNISTImages('train-images.idx3-ubyte');
label = loadMNISTLables('train-labels.idx1-ubyte');
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_label = loadMNISTLables('t10k-labels.idx1-ubyte');

eta = 0.1;

v_ih = rand(784,30)./10000;
w_hj = rand(30, 10)./10000;
b_h = zeros(30,1);
gamma_h = rand(30,1);
theta_j = rand(10,1);
e_h = rand(30,1);

for i = 1 : 60000
   image = images(:,i);
   alpha_h = neu_cal(image, v_ih); %\alpha_h
   b_h = sigmoid(alpha_h, gamma_h);
   beta_j = neu_cal_out(b_h, w_hj); %\beta_j
   result = sigmoid(beta_j, theta_j); %\hat{y_j^k}
   label_y = zeros(10,1);
   label_y(label(i)+1) = 1;
   g_j = result.*(1-result).*(label_y-result);
   e_h = b_h.*(1-b_h).*(w_hj*g_j);
   
   for h = 1:30
       for j = 1:10
           w_hj(h,j) = w_hj(h,j) + eta*g_j(j) * b_h(h);
       end
   end
   
   for j = 1:10
       theta_j(j) = theta_j(j) - eta*g_j(j);
   end
   
   for i = 1:784
       for h = 1:30
           v_ih(i,h) = v_ih(i,h)+ eta*e_h(h)*image(i);
       end
   end
   
   for h = 1:30
       gamma_h(h) = gamma_h(h)-eta*e_h(h);
   end
end

correct_num = 0;

for i = 1:10000
test_image = test_images(:,i);
alpha_h = neu_cal(test_image, v_ih); %\alpha_h
b_h = sigmoid(alpha_h, gamma_h);
beta_j = neu_cal_out(b_h, w_hj); %\beta_j
result = sigmoid(beta_j, theta_j) %\hat{y_j^k}

number = find(result == max(max(result))) - 1;
if number == test_label(i)
    correct_num = correct_num + 1;
end

end

correct_num
correct_num / 10000

function neu = neu_cal(image, neu_12)
    neu = (image'*neu_12)';
end

function output = neu_cal_out(neu, neu_23)
    output = (neu'*neu_23)';
end

function result = sigmoid(beta, theta)
    result = 1./(1+exp(-(beta-theta)));
end
