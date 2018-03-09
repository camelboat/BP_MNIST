clear all;

images = loadMNISTImages('train-images.idx3-ubyte');
label = loadMNISTLables('train-labels.idx1-ubyte');
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_label = loadMNISTLables('t10k-labels.idx1-ubyte');

% set up your learning rate here
eta = 0.2;

% set up your training times here
training_times = 10;

v_ih = rand(784,30)/10000;
%w_hj = rand(30, 10)/10000;

%v_ih = zeros(784, 30);
w_hj = zeros(30, 10);

b_h = zeros(30,1);
gamma_h = rand(30,1);
theta_j = rand(10,1);
e_h = rand(30,1);

disp('start training: ');

for time = 1:training_times
    disp([num2str(time), ' time of taining']);
    for i = 1 : 60000
       if mod(i,600) == 0
           disp([num2str(i/600), '%']);
       end

       image = images(:,i);
       alpha_h = neu_cal(image, v_ih); %\alpha_h
       b_h = sigmoid(alpha_h, gamma_h);
       beta_j = neu_cal(b_h, w_hj); %\beta_j
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
end

disp(['training complete!']);
disp('start to validate')

correct_num = 0;

for i = 1:10000
    test_image = test_images(:,i);
    alpha_h = neu_cal(test_image, v_ih); %\alpha_h
    b_h = sigmoid(alpha_h, gamma_h);
    beta_j = neu_cal(b_h, w_hj); %\beta_j
    result = sigmoid(beta_j, theta_j); %\hat{y_j^k}

    number = find(result == max(max(result))) - 1;

    if number == test_label(i)
        correct_num = correct_num + 1;
    end
end

accuracy = (correct_num / 10000) * 100;
error_rate = 100 - accuracy;

disp(['Total correct number is: ', num2str(correct_num)]);
disp(['correct rate is: ', num2str(accuracy), '%']);
disp(['error tate is: ', num2str(error_rate), '%']);


