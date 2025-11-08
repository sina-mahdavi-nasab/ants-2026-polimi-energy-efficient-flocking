fid = fopen('signals.par','w');
fprintf(fid,'stop outcmaes\n');   % hard-coded prefix
fclose(fid);