python -W ignore lstm_wo.py > out_lstm_wo.txt
for adp in 1 2 3 ; 
do 
python -W ignore lstm_nsteps_adaptive.py ${adp} > out_lstm_adaptive_${adp}.txt
done
for steps in 1 2 5 10 ;
do
python -W ignore lstm_nsteps.py ${steps} >  out_lstm_nstep_${steps}.txt
done
