#!/bin/bash

BASEDIR=`dirname $0`/..
BASEDIR=`(cd "$BASEDIR" ; pwd)`
BASEDIR_GSARF=`dirname $0`/../..
BASEDIR_GSARF=`(cd "$BASEDIR_GSARF" ; pwd)`

MEMORY=512m

for seed in {0..9} ; do
java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2018.6.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
"WriteStreamToARFFFile -s (RecurrentConceptDriftStream -x 50000 -y 100000 -z 10 -s (generators.AgrawalGenerator -i $seed) -d (generators.AgrawalGenerator -f 2 -i $seed) -p 50000 -w 1) -f $BASEDIR_GSARF/data/agrawal_2_w1/$seed.csv -m 1000000 -h"
done

for seed in {0..9} ; do
java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2018.6.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
"WriteStreamToARFFFile -s (RecurrentConceptDriftStream -x 50000 -y 100000 -z 10 -s (generators.AgrawalGenerator -i $seed) -d (generators.AgrawalGenerator -f 2 -i $seed) -p 50000 -w 25000) -f $BASEDIR_GSARF/data/agrawal_2_w25k/$seed.csv -m 1000000 -h"
done

for seed in {0..9} ; do
java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2018.6.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
"WriteStreamToARFFFile -s (RecurrentConceptDriftStream -x 50000 -y 100000 -z 10 -s (generators.AgrawalGenerator -i $seed) -d (generators.AgrawalGenerator -f 2 -i $seed) -p 50000 -w 50000) -f $BASEDIR_GSARF/data/agrawal_2_w50k/$seed.csv -m 1000000 -h"
done

for seed in {0..9} ; do
java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2018.6.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
"WriteStreamToARFFFile -s (RecurrentConceptDriftStream -x 200000 -y 100000 -z 10 -s (generators.AgrawalGenerator -i $seed) -d (RecurrentConceptDriftStream -x 100000 -y 50000 -z 10 -s (generators.AgrawalGenerator -f 7 -i $seed) -d (generators.AgrawalGenerator -f 4 -i $seed) -p 0 -w 25000) -p 100000 -w 1) -f $BASEDIR_GSARF/data/agrawal_3_mixed/$seed.csv -m 1000000 -h"
done

for seed in {0..9} ; do
java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2018.6.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
"WriteStreamToARFFFile -s (RecurrentConceptDriftStream -x 200000 -y 100000 -z 10 -s (generators.AgrawalGenerator -i $seed) -d (RecurrentConceptDriftStream -x 100000 -y 50000 -z 10 -s (generators.AgrawalGenerator -f 7 -i $seed) -d (generators.AgrawalGenerator -f 4 -i $seed) -p 0 -w 1) -p 100000 -w 1) -f $BASEDIR_GSARF/data/agrawal_3_w1/${seed}.csv -m 1000000 -h"
done

for seed in {0..9} ; do
java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2018.6.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
"WriteStreamToARFFFile -s (RecurrentConceptDriftStream -x 200000 -y 100000 -z 10 -s (generators.AgrawalGenerator -i $seed) -d (RecurrentConceptDriftStream -x 100000 -y 50000 -z 10 -s (generators.AgrawalGenerator -f 7 -i $seed) -d (generators.AgrawalGenerator -f 4 -i $seed) -p 0 -w 25000) -p 100000 -w 25000) -f $BASEDIR_GSARF/data/agrawal_3_w25k/${seed}.csv -m 1000000 -h"
done

# case study
java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2018.6.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
"WriteStreamToARFFFile -s (RecurrentConceptDriftStream -x 500000 -y 200000 -z 10 -s (generators.AgrawalGenerator -f 1) -d (RecurrentConceptDriftStream -x 100000 -y 200000 -z 10 -s (generators.AgrawalGenerator -f 7) -d (generators.AgrawalGenerator -f 4) -p 100000 -w 100000) -p 100000 -w 100000) -f $BASEDIR_GSARF/data/agrawal_case_study/3m.csv -m 3000000 -h"

function spin {
	pid=$!
	
	spin='-\|/'
	
	i=0
	while kill -0 $pid 2>/dev/null ; do
	  i=$(( (i+1) %4 ))
	  printf "\rbinning $1 ... ${spin:$i:1}"
	  sleep .1
	done

	echo -ne "\rbinning for $1 completed"
}


for drift_type in 2_w1 2_w25k 2_w50k 3_w1 3_w25k 3_mixed ; do
	for seed in {0..9} ; do
		./converter.py $BASEDIR_GSARF/data/agrawal_$drift_type/$seed.csv &
		echo 
		spin $BASEDIR_GSARF/data/agrawal_$drift_type/$seed.csv
	done
done

./converter.py $BASEDIR_GSARF/data/agrawal_case_study/3m.csv &
echo 
spin $BASEDIR_GSARF/data/agrawal/agrawal_case_study/3m.csv
