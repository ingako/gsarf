#!/bin/bash

BASEDIR=`dirname $0`/..
BASEDIR=`(cd "$BASEDIR"; pwd)`
BASEDIR_GSARF=`dirname $0`/../..
BASEDIR_GSARF=`(cd "$BASEDIR_GSARF" ; pwd)`
MEMORY=10024m

for drift_type in 2_w1 2_w25 2_w50 3_w1 3_w25 3_mixed ; do
	for seed in {0..9} ; do

# adwin drift delta is default 0.00001
java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2018.6.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
"EvaluatePrequential -l (meta.AdaptiveRandomForest -l (ARFHoeffdingTree -k 6 -e 2000000 -g 50 -c 0.05 -t 0.0 -b -p -l MC) -s 60 -a 1.0 -p (ADWINChangeDetector -a 0.001) -w) -s (clustering.SimpleCSVStream -f $BASEDIR_GSARF/data/led_${drift_type}/${seed}.csv -c) -f 1000 -d $BASEDIR_GSARF/data/led_${drift_type}/result_moa_${seed}.csv"

	done
done


