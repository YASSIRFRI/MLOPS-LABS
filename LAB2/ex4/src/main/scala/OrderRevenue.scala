import scala.io.Source
object OrderRevenue {
def main(args: Array[String]) = {
val orderId = args(1).toInt
val orderItems = Source.fromFile("part-00000").getLines
val orderRevenue = orderItems.filter(oi => oi.split(",")(1).toInt ==
orderId).
map(oi => oi.split(",")(4).toFloat).
reduce((t, v) => t + v)
println(orderRevenue)
}
}