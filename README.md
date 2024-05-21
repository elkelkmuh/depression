clc,clear all,close all
dosya=dir('*.wav');
for k=1:length(dosya)
    k
    [sinyal,fs]=audioread(dosya(k).name);
    indices(k)=dosya(k).name(3:4);
    tt=sinyal(:,1);
    X(k,1:296)=[ist40(tt) arcslbp(tt)];
    y(k)=str2num(dosya(k).name(1));
    say=296;
    for i=1:5
        [low,high]=dwt(tt,'sym4');
        X(k,say+1:say+592)=[arcslbp(low) ist40(low) ist40(high) arcslbp(high)];
        say=say+592;
        clear tt
        tt=low;
        clear low high
    end
    clear sinyal fs
 end
[mm,nn]=size(X);
for i=1:nn
    X(:,i)=(X(:,i)-min(X(:,i)))/(max(X(:,i))-min(X(:,i))+eps);
end
mdl=fscnca(X,y,'Solver','sgd','Verbose',1);
xx=mdl.FeatureWeights;
[aa,ind]=sort(xx,'desc');
for ts=0:900
    for i=1:100+ts
        poz(:,i)=X(:,ind(i));
    end
    mdl = fitcknn(...
    poz, ...
    y, ...
    'Distance', 'Cityblock', ...
    'Exponent', [], ...
    'NumNeighbors', 5, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [1:40]');
    kk=crossval(mdl,'KFold',5);
    ll(ts+1) = kfoldLoss(kk);
    clear poz
    end
    [eb,inde]=min(ll);
    for i=1:inde+99
        son(:,i)=X(:,ind(i));
    end
    son(:,i+1)=y; 

    
