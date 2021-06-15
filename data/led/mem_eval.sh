#!/usr/bin/env bash
awk '{ total += $2 } END { print total/NR }' log.tree_pool_size

awk 'function sdev(array) {
     for (i=1; i in array; i++)
        sum+=array[i]
     cnt=i-1
     mean=sum/cnt
     for (i=1; i in array; i++)
        sqdif+=(array[i]-mean)**2
     return (sqdif/(cnt-1))**0.5
     }
     {sum1[FNR]=$2}
     END {print sdev(sum1)}' log.tree_pool_size
