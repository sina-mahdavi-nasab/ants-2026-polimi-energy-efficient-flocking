clc
clear

load('outcmaesxrecentbest.dat')

[~,bestgen] = min(outcmaesxrecentbest(:,3)) ;
% ABCD_mat = bestABCD;
% rules = unflattenABCD(ABCD_mat);
ABCD_mat = outcmaesxrecentbest(bestgen,6:end)';
ABCD_mat = max(ABCD_mat,-5);
ABCD_mat = min(ABCD_mat,5);


% load("bestABCD(6)_Collision20_3Wall");
% ABCD_mat = bestABCD;

rules = unflattenABCD(ABCD_mat);
seed = 1;
tic;
% [plotting, distance_travelled, avg_batt, collsions] = simulation_free_global_mod_2_peter(rules,seed);
simulation_free_global_mod_2(rules,seed);
toc;

function R = unflattenABCD(v)
    idx=0; take=@(n) reshape(v(idx+(1:n)),[],1);
    grab=@(r,c) reshape( take(r*c), r,c );
    R.A1 = grab(10,10); idx=idx+100;
    R.A2 = grab(10,10); idx=idx+100;
    R.A3 = grab(10, 2); idx=idx+20 ;
    R.B1 = grab(10,10); idx=idx+100;
    R.B2 = grab(10,10); idx=idx+100;
    R.B3 = grab(10, 2); idx=idx+20 ;
    R.C1 = grab(10,10); idx=idx+100;
    R.C2 = grab(10,10); idx=idx+100;
    R.C3 = grab(10, 2); idx=idx+20 ;
    R.D1 = grab(10,10); idx=idx+100;
    R.D2 = grab(10,10); idx=idx+100;
    R.D3 = grab(10, 2);

end
