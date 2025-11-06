



val v = (1 to 100).toList

val pair=v.filter(i=> i%2==0)


val squared =pair.map(i=>i*i)


val somme= squared.reduce((i,next)=> i+next)


println(somme)