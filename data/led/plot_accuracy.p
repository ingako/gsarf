set datafile separator ","
set terminal qt size 1000,600

plot 'result_moa_test.csv' using 1:5 title 'moa' with lines linestyle 1,\
'result_gpu_bg_led_gradual_test.csv' using 1:2 title 'gpu-bg' with lines linestyle 2,\
'result_gpu_led_gradual_test.csv' using 1:2 title 'gpu' with lines linestyle 3
