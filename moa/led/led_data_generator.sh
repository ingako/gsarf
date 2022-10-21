#!/bin/bash

BASEDIR=`dirname $0`/..
BASEDIR=`(cd "$BASEDIR" ; pwd)`
BASEDIR_GSARF=`dirname $0`/../..
BASEDIR_GSARF=`(cd "$BASEDIR_GSARF" ; pwd)`

MEMORY=512m

for seed in {0..9} ; do
java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2018.6.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
"WriteStreamToARFFFile -s (RecurrentConceptDriftStream -x 50000 -y 100000 -z 10 -s (generators.LEDGeneratorDrift -i $seed) -d (generators.LEDGeneratorDrift -d 6 -i $seed) -p 50000 -w 1) -f $BASEDIR_GSARF/data/led_2_w1/$seed.csv -m 1000000 -h"
done

for seed in {0..9}; do
java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2018.6.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
"WriteStreamToARFFFile -s (RecurrentConceptDriftStream -x 50000 -y 100000 -z 10 -s (generators.LEDGeneratorDrift -i ${seed}) -d (generators.LEDGeneratorDrift -d 6 -i ${seed}) -p 50000 -w 25000) -f $BASEDIR_GSARF/data/led_2_w25k/${seed}.csv -m 1000000 -h"
done

for seed in {0..9} ; do
java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2018.6.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
"WriteStreamToARFFFile -s (RecurrentConceptDriftStream -x 50000 -y 100000 -z 10 -s (generators.LEDGeneratorDrift -i $seed) -d (generators.LEDGeneratorDrift -d 6 -i $seed) -p 50000 -w 50000) -f $BASEDIR_GSARF/data/led_2_w50k/${seed}.csv -m 1000000 -h"
done

for seed in {0..9} ; do
java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2018.6.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
"WriteStreamToARFFFile -s (RecurrentConceptDriftStream -x 200000 -y 100000 -z 10 -s (generators.LEDGeneratorDrift -i $seed) -d (RecurrentConceptDriftStream -x 100000 -y 50000 -z 10 -s (generators.LEDGeneratorDrift -d 7 -i $seed) -d (generators.LEDGeneratorDrift -d 4 -i $seed) -p 0 -w 25000) -p 100000 -w 1) -f $BASEDIR_GSARF/data/led_3_mixed/${seed}.csv -m 1000000 -h"
done

for seed in {0..9} ; do
java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2018.6.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
"WriteStreamToARFFFile -s (RecurrentConceptDriftStream -x 200000 -y 100000 -z 10 -s (generators.LEDGeneratorDrift -i $seed) -d (RecurrentConceptDriftStream -x 100000 -y 50000 -z 10 -s (generators.LEDGeneratorDrift -d 7 -i $seed) -d (generators.LEDGeneratorDrift -d 4 -i $seed) -p 0 -w 1) -p 100000 -w 1) -f $BASEDIR_GSARF/data/led_3_w1/${seed}.csv -m 1000000 -h"
done

for seed in {0..9} ; do
java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2018.6.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
"WriteStreamToARFFFile -s (RecurrentConceptDriftStream -x 200000 -y 100000 -z 10 -s (generators.LEDGeneratorDrift -i $seed) -d (RecurrentConceptDriftStream -x 100000 -y 50000 -z 10 -s (generators.LEDGeneratorDrift -d 7 -i $seed) -d (generators.LEDGeneratorDrift -d 4 -i $seed) -p 0 -w 25000) -p 100000 -w 25000) -f $BASEDIR_GSARF/data/led_3_w25k/${seed}.csv -m 1000000 -h"
done
