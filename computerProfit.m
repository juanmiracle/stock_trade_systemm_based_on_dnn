load('parameter.mat');
patha = 'test\week';
pathb = '.csv';
pb = 1.0;
pa = 1.0;
ps = 1.0;
pt =1.0;
pk = 1.0;
T = 300;
%L = 5;
pq = [];
for j = 1:40
    if(j>=100)
        k1 = mod(j,10);
        k2 = mod((j-k1)/10,10);
        k3 = (j-k2*10-k1)/100;
        fprintf(['week' 48+k3 48+k2 48+k1 '\n']);
        testdata = csvread([patha 48+k3 48+k2 48+k1 pathb]); 
    else if(j>=10)
        k1 = mod(j,10);
        k2 = (j-k1)/10;
        fprintf(['week' 48+k2 48+k1 '\n']);
        testdata = csvread([patha 48+k2 48+k1 pathb]); 
        else
          fprintf(['week' 48+j '\n']);
           testdata = csvread([patha 48+j pathb]); 
        end
    end
    testdata = testdata';
    testData =1 ./ (1 + exp(-alpha.*testdata(1:80,:)));
    profit = testdata(81,:);
   [bpred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, numClasses, netconfig, testData);
   [num,val] = sort(bpred);
    pf = 0.0;
    for i=T+1-L:T
     pf=pf+profit(val(i));
    end
    pf=pf/L;
    pb = pb*(1+pf);
    fprintf('Before Finetuning profit:%0.3f%%\n',pb*100-100);
    
   [pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, numClasses, netconfig, testData);
   [num,val] = sort(pred);
    pf = 0.0;
    for i=T+1-L:T
     pf=pf+profit(val(i));
    end
    pf=pf/L;
    pa = pa*(1+pf);
    pq=[pq pa*100-100];
    fprintf('After Finetuning profit:%0.3f%%\n',pa*100-100);
    %% softmax
    [spred] = softmaxPredict(softmaxModel, testData);
    [num,val] = sort(spred);
    pf = 0.0;
    for i=T+1-L:T
     pf=pf+profit(val(i));
    end
    pf=pf/L;
    ps = ps*(1+pf);
    fprintf('Softmax Test profit:%0.3f%%\n',ps*100-100);
    
    [num,val1] = sort(pred);
    [num,val2] = sort(spred);
    K=0;
    pf = 0.0;
    pr=[];
    for i = T-20:T
        if ismember(val1(i),val2(T-20:T))
            pf=pf+profit(val1(i));
            K=K+1;
            pr = [pr val1(i)];
        end
    end
    if K>0
        pf = pf/K;
    else
        pf = 0;
    end
    pt = pt*(1+pf);
    fprintf('Softmax Test profit:%0.3f%%\n',pt*100-100);
    
    t = pred+0.3.*spred;
    [num,val] = sort(t);
    pf = 0.0;
    for i=T+1-L:T
     pf=pf+profit(val(i));
    end
    pf=pf/L;
    pk = pk*(1+pf);
    fprintf('Softmax Test profit:%0.3f%%\n',pk*100-100);
end