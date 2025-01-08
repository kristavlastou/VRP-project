from Model import *
from SolutionDrawer import *
import random
import heapq


class Solution:
    def __init__(self):
        self.cost = 0.0
        self.routes = []


class RelocationMove(object):
    def __init__(self):
        self.originRoutePosition = None
        self.targetRoutePosition = None
        self.originNodePosition = None
        self.targetNodePosition = None
        self.costChangeOriginRt = None
        self.costChangeTargetRt = None
        self.moveCost = None
        # self.moveCost_penalized = None

    def Initialize(self):
        self.originRoutePosition = None
        self.targetRoutePosition = None
        self.originNodePosition = None
        self.targetNodePosition = None
        self.costChangeOriginRt = None
        self.costChangeTargetRt = None
        self.moveCost = 10 ** 9
        # self.moveCost_penalized = 10 ** 9


class SwapMove(object):
    def __init__(self):
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.costChangeFirstRt = None
        self.costChangeSecondRt = None
        self.moveCost = None
        # self.moveCost_penalized = None

    def Initialize(self):
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.costChangeFirstRt = None
        self.costChangeSecondRt = None
        self.moveCost = 10 ** 9
        # self.moveCost_penalized = 10 ** 9


class TwoOptMove(object):
    def __init__(self):
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.moveCost = None
        # self.moveCost_penalized = None

    def Initialize(self):
        self.positionOfFirstRoute = None
        self.positionOfSecondRoute = None
        self.positionOfFirstNode = None
        self.positionOfSecondNode = None
        self.moveCost = 10 ** 9
        # self.moveCost_penalized = 10 ** 9


class CustomerInsertion(object):
    def __init__(self):
        self.customer = None
        self.route = None
        self.cost = 10 ** 9


class CustomerInsertionAllPositions(object):
    def __init__(self):
        self.customer = None
        self.route = None
        self.insertionPosition = None
        self.cost = 10 ** 9


class Solver:
    def __init__(self, m):
        self.allNodes = m.allNodes
        self.customers = m.customers
        self.depot = m.allNodes[0]
        self.distance_matrix = m.matrix
        self.capacity = m.capacity
        self.sol = None
        self.bestSolution = None
        rows = len(self.allNodes)
        # self.distance_matrix_penalized = [[self.distance_matrix[i][j] for j in range(rows)] for i in range(rows)]
        # self.times_penalized = [[0 for j in range(rows)] for i in range(rows)]
        # self.penalized_n1_ID = -1
        # self.penalized_n2_ID = -1

    def calculate_route_details(self, nodes_sequence, empty_vehicle_weight):
        tot_dem = sum(n.demand for n in nodes_sequence)
        tot_load = empty_vehicle_weight + tot_dem
        tn_km = 0
        for i in range(len(nodes_sequence) - 1):
            from_node = nodes_sequence[i]
            to_node = nodes_sequence[i + 1]
            tn_km += self.distance_matrix[from_node.ID][to_node.ID] * tot_load
            tot_load -= to_node.demand
        return tn_km, tot_dem

    def VND(self):
        self.bestSolution = self.cloneSolution(self.sol)
        VNDIterator = 0
        kmax = 2
        rm = RelocationMove()
        sm = SwapMove()
        top = TwoOptMove()
        k = 0

        while k <= kmax:
            self.InitializeOperators(rm, sm, top)
            if k == 2:
                self.FindBestRelocationMove(rm)
                if rm.originRoutePosition is not None and rm.moveCost < 0:
                    self.ApplyRelocationMove(rm)

                    VNDIterator = VNDIterator + 1

                    k = 0
                else:
                    k += 1
            elif k == 1:
                self.FindBestSwapMove(sm)
                if sm.positionOfFirstRoute is not None and sm.moveCost < 0:
                    self.ApplySwapMove(sm)

                    VNDIterator = VNDIterator + 1

                    k = 0
                else:
                    k += 1
            elif k == 0:
                self.FindBestTwoOptMove(top)
                if top.positionOfFirstRoute is not None and top.moveCost < 0:
                    self.ApplyTwoOptMove(top)

                    VNDIterator = VNDIterator + 1

                    k = 0
                else:
                    k += 1

            if self.sol.cost < self.bestSolution.cost:
                self.bestSolution = self.cloneSolution(self.sol)

    def calc_tonne_kilometre_route(self, rt):
        nodes_sequence = rt.sequenceOfNodes
        tot_dem = sum(n.demand for n in nodes_sequence)
        tot_load = 8 + tot_dem
        tn_km = 0
        for i in range(len(nodes_sequence) - 1):
            from_node = nodes_sequence[i]
            to_node = nodes_sequence[i + 1]
            tn_km += self.distance_matrix[from_node.ID][to_node.ID] * tot_load
            tot_load -= to_node.demand
        return tn_km

    def calc_tonne_kilometre_total(self, sol):
        solution_tonne_kilometre_sum = 0
        for rt in sol.routes:
            route_sum = self.calc_tonne_kilometre_route(rt)
            solution_tonne_kilometre_sum += route_sum
        sol.cost = solution_tonne_kilometre_sum
        return solution_tonne_kilometre_sum

    def solve(self):
        self.SetRoutedFlagToFalseForAllCustomers()
        self.sol = Solution()
        self.ApplyNearestNeighborMethod()
        self.LocalSearch(2)
        self.VND()
        return self.sol

    def SetRoutedFlagToFalseForAllCustomers(self):
        for i in range(0, len(self.customers)):
            self.customers[i].isRouted = False

    def ApplyNearestNeighborMethod(self):
        modelIsFeasible = True
        self.sol = Solution()
        insertions = 0

        while (insertions < len(self.customers)):
            bestInsertion = CustomerInsertion()
            lastOpenRoute: Route = self.GetLastOpenRoute()

            if lastOpenRoute is not None:
                self.IdentifyBestInsertion(bestInsertion, lastOpenRoute)

            if (bestInsertion.customer is not None):
                self.ApplyCustomerInsertion(bestInsertion)
                insertions += 1
            else:
                # If there is an empty available route
                if lastOpenRoute is not None and len(lastOpenRoute.sequenceOfNodes) == 2:
                    modelIsFeasible = False
                    break
                else:
                    rt = Route(self.depot, self.capacity)
                    self.sol.routes.append(rt)

        if (modelIsFeasible == False):
            print('FeasibilityIssue')
            # reportSolution

    def Always_keep_an_empty_route(self):
        if len(self.sol.routes) == 0:
            rt = Route(self.depot, self.capacity)
            self.sol.routes.append(rt)
        else:
            rt = self.sol.routes[-1]
            if len(rt.sequenceOfNodes) > 2:
                rt = Route(self.depot, self.capacity)
                self.sol.routes.append(rt)

    def MinimumInsertionsStochastic(self):
        """
        Constructs an initial solution using stochastic minimum insertions.
        """
        model_is_feasible = True
        self.sol = Solution()
        insertions = 0

        while insertions < len(self.customers):
            # Always keep an empty route
            self.Always_keep_an_empty_route()

            # Gather all insertion candidates
            candidates = self.IdentifyMinimumCostInsertionCandidates()

            # Select the best insertion probabilistically
            best_insertion = self.IdentifyMinimumCostInsertionStochastic(candidates)

            if best_insertion is not None:
                self.ApplyCustomerInsertionAllPositions(best_insertion)
                insertions += 1
            else:
                print('FeasibilityIssue')
                model_is_feasible = False
                break

        if model_is_feasible:
            self.TestSolution()

    def IdentifyMinimumCostInsertionCandidates(self):
        """
        Gathers all possible insertions and their costs.
        Returns a list of tuples: (customer, route, position, cost).
        """
        candidates = []
        for i in range(0, len(self.customers)):
            candidateCust: Node = self.customers[i]
            if candidateCust.isRouted is False:
                for rt in self.sol.routes:
                    if rt.load + candidateCust.demand <= rt.capacity:
                        for j in range(0, len(rt.sequenceOfNodes) - 1):
                            A = rt.sequenceOfNodes[j]
                            B = rt.sequenceOfNodes[j + 1]
                            costAdded = self.distance_matrix[A.ID][candidateCust.ID] + \
                                        self.distance_matrix[candidateCust.ID][B.ID]
                            costRemoved = self.distance_matrix[A.ID][B.ID]
                            trialCost = costAdded - costRemoved
                            candidates.append((candidateCust, rt, j, trialCost))
                    else:
                        continue
        return candidates

    def IdentifyMinimumCostInsertionStochastic(self, candidate_insertions):
        """
        Stochastic approach for the Minimum Insertions algorithm
        Identifies a customer for insertion probabilistically based on cost
        candidate_insertions: List of (customer, route, position, cost)
        Returns a selected CustomerInsertionAllPositions object that represents the best insertion
        """
        random.seed(1)
        # No candidate insertions --> no feasible solution
        if not candidate_insertions:
            return None

        # Extract costs from matrix
        costs = [insertion[3] for insertion in candidate_insertions]

        # Normalize costs to probabilities (lower cost --> higher probability)
        max_cost = max(costs)
        exp_costs = [math.exp(-(cost - max_cost)) for cost in costs]
        total = sum(exp_costs)
        probabilities = [exp_cost / total for exp_cost in exp_costs]

        # Select a candidate based on probabilities
        selected_index = random.choices(range(len(candidate_insertions)), probabilities)[0]
        selected_insertion = candidate_insertions[selected_index]

        # Convert back to a CustomerInsertionAllPositions object
        best_insertion = CustomerInsertionAllPositions()
        best_insertion.customer = selected_insertion[0]
        best_insertion.route = selected_insertion[1]
        best_insertion.insertionPosition = selected_insertion[2]
        best_insertion.cost = selected_insertion[3]
        return best_insertion

    def LocalSearch(self, operator):
        random.seed(1)
        self.bestSolution = self.cloneSolution(self.sol)
        terminationCondition = False
        localSearchIterator = 0

        rm = RelocationMove()
        sm = SwapMove()
        top = TwoOptMove()

        while terminationCondition is False:
            operator = random.randint(0, 2)
            self.InitializeOperators(rm, sm, top)
            # SolDrawer.draw(localSearchIterator, self.sol, self.allNodes)
            # Relocations
            if operator == 0:
                self.FindBestRelocationMove(rm)
                if rm.originRoutePosition is not None:
                    # if rm.moveCost_penalized < 0:
                    if rm.moveCost < 0:
                        self.ApplyRelocationMove(rm)
                        # print(localSearchIterator, self.sol.cost, operator)
                    else:
                        # self.penalize_arcs()
                        localSearchIterator = localSearchIterator - 1
            # Swaps
            elif operator == 1:
                self.FindBestSwapMove(sm)
                if sm.positionOfFirstRoute is not None:
                    # if sm.moveCost_penalized < 0:
                    if sm.moveCost < 0:
                        self.ApplySwapMove(sm)
                        # print(localSearchIterator, self.sol.cost, operator)
                    else:
                        # self.penalize_arcs()
                        localSearchIterator = localSearchIterator - 1
            elif operator == 2:
                self.FindBestTwoOptMove(top)
                if top.positionOfFirstRoute is not None:
                    # if top.moveCost_penalized < 0:
                    if top.moveCost < 0:
                        self.ApplyTwoOptMove(top)
                        # print(localSearchIterator, self.sol.cost, operator)
                    else:
                        # self.penalize_arcs()
                        localSearchIterator = localSearchIterator - 1

            self.TestSolution()

            if (self.sol.cost < self.bestSolution.cost):
                self.bestSolution = self.cloneSolution(self.sol)
                print(localSearchIterator, self.bestSolution.cost)

            localSearchIterator = localSearchIterator + 1
            if localSearchIterator == 1000:
                terminationCondition = True

        self.sol = self.bestSolution
        SolDrawer.draw(self.bestSolution.cost, self.sol, self.allNodes)

    def cloneRoute(self, rt: Route):
        cloned = Route(self.depot, self.capacity)
        cloned.cost = rt.cost
        cloned.load = rt.load
        cloned.sequenceOfNodes = rt.sequenceOfNodes.copy()
        return cloned

    def cloneSolution(self, sol: Solution):
        cloned = Solution()
        for i in range(0, len(sol.routes)):
            rt = sol.routes[i]
            clonedRoute = self.cloneRoute(rt)
            cloned.routes.append(clonedRoute)
        cloned.cost = self.sol.cost
        return cloned

    # with penalized and target/origin
    def FindBestRelocationMove(self, rm):
        for originRouteIndex in range(0, len(self.sol.routes)):
            rt1: Route = self.sol.routes[originRouteIndex]
            for originNodeIndex in range(1, len(rt1.sequenceOfNodes) - 1):
                for targetRouteIndex in range(0, len(self.sol.routes)):
                    rt2: Route = self.sol.routes[targetRouteIndex]
                    for targetNodeIndex in range(0, len(rt2.sequenceOfNodes) - 1):

                        if originRouteIndex == targetRouteIndex and (
                                targetNodeIndex == originNodeIndex or targetNodeIndex == originNodeIndex - 1):
                            continue

                        A = rt1.sequenceOfNodes[originNodeIndex - 1]
                        B = rt1.sequenceOfNodes[originNodeIndex]
                        C = rt1.sequenceOfNodes[originNodeIndex + 1]

                        F = rt2.sequenceOfNodes[targetNodeIndex]
                        G = rt2.sequenceOfNodes[targetNodeIndex + 1]

                        if rt1 != rt2:
                            if rt2.load + B.demand > rt2.capacity:
                                continue

                        d_after_ornode = sum(node.demand for node in rt1.sequenceOfNodes[originNodeIndex + 1:])
                        d_after_tarnode = sum(node.demand for node in rt2.sequenceOfNodes[targetNodeIndex + 1:])

                        if rt1 == rt2:
                            cost_added = 0
                            cost_removed = 0

                            # cost_added_penalized = 0
                            # cost_removed_penalized = 0

                            if originNodeIndex < targetNodeIndex:
                                # calculates the extra cost that occurs from the demand of the node
                                for i in range(originNodeIndex + 1, targetNodeIndex):
                                    nextnode = i + 1
                                    n1 = rt2.sequenceOfNodes[i]
                                    n2 = rt2.sequenceOfNodes[nextnode]
                                    cost_added += B.demand * self.distance_matrix[n1.ID][n2.ID]
                                    # cost_added_penalized += B.demand * self.distance_matrix[n1.ID][n2.ID]
                                # T=10
                                cost_added += self.distance_matrix[A.ID][C.ID] * (d_after_ornode + B.demand + 8) + \
                                              self.distance_matrix[F.ID][B.ID] * (d_after_tarnode + B.demand + 8) + \
                                              self.distance_matrix[B.ID][G.ID] * (d_after_tarnode + 8)
                                cost_removed = self.distance_matrix[A.ID][B.ID] * (d_after_ornode + B.demand + 8) - \
                                               self.distance_matrix[B.ID][C.ID] * (d_after_ornode + 8) - \
                                               self.distance_matrix[F.ID][G.ID] * (d_after_tarnode + 8)

                                # cost_added_penalized += self.distance_matrix_penalized[A.ID][C.ID] * (d_after_ornode + B.demand + 8)+ \
                                #     self.distance_matrix_penalized[F.ID][B.ID] * (d_after_tarnode + B.demand + 8) + \
                                #     self.distance_matrix_penalized[B.ID][G.ID] * (d_after_tarnode + 8)
                                # cost_removed_penalized = self.distance_matrix_penalized[A.ID][B.ID] * (d_after_ornode + B.demand + 8) - \
                                #     self.distance_matrix_penalized[B.ID][C.ID] * (d_after_ornode + 8) - \
                                #     self.distance_matrix_penalized[F.ID][G.ID] * (d_after_tarnode + 8)




                            # if targetNodeIndex< originNodeIndex
                            else:
                                # calculates the removed cost that occurs from the fact that the node leaves the route earlier
                                for i in range(targetNodeIndex + 1, originNodeIndex - 1):
                                    nextnode = i + 1
                                    n1 = rt2.sequenceOfNodes[i]
                                    n2 = rt2.sequenceOfNodes[nextnode]
                                    cost_removed += B.demand * self.distance_matrix[n1.ID][n2.ID]
                                    # cost_removed_penalized += B.demand * self.distance_matrix[n1.ID][n2.ID]

                                # T=10
                                cost_added = self.distance_matrix[A.ID][C.ID] * (d_after_ornode + 8) + \
                                             self.distance_matrix[F.ID][B.ID] * (d_after_tarnode + 8) + \
                                             self.distance_matrix[B.ID][G.ID] * (d_after_tarnode - B.demand + 8)
                                cost_removed += self.distance_matrix[A.ID][B.ID] * (d_after_ornode + B.demand + 8) - \
                                                self.distance_matrix[B.ID][C.ID] * (d_after_ornode + 8) - \
                                                self.distance_matrix[F.ID][G.ID] * (d_after_tarnode + 8)

                                # cost_added_penalized = self.distance_matrix_penalized[A.ID][C.ID] * (d_after_ornode + 8)+ \
                                #     self.distance_matrix_penalized[F.ID][B.ID] * (d_after_tarnode +  8) + \
                                #     self.distance_matrix_penalized[B.ID][G.ID] * (d_after_tarnode - B.demand + 8)
                                # cost_removed_penalized += self.distance_matrix_penalized[A.ID][B.ID] * (d_after_ornode + B.demand + 8) - \
                                #     self.distance_matrix_penalized[B.ID][C.ID] * (d_after_ornode + 8) - \
                                #     self.distance_matrix_penalized[F.ID][G.ID] * (d_after_tarnode + 8)

                            moveCost = cost_added - cost_removed

                            originRtCostChange = moveCost
                            targetRtCostChange = moveCost

                            # moveCost_penalized = cost_added_penalized - cost_removed_penalized
                        # if rt1 != rt2
                        else:
                            cost_rt1 = 0
                            cost_rt2 = 0
                            # cost_rt1_penalized = 0
                            # cost_rt2_penalized = 0

                            for i in range(0, originNodeIndex - 1):
                                nextnode = i + 1
                                n1 = rt1.sequenceOfNodes[i]
                                n2 = rt1.sequenceOfNodes[nextnode]
                                cost_rt1 -= B.demand * self.distance_matrix[n1.ID][n2.ID]
                                # cost_rt1_penalized -= B.demand * self.distance_matrix[n1.ID][n2.ID]

                            cost_rt1 += self.distance_matrix[A.ID][C.ID] * (d_after_ornode + 8) - \
                                        self.distance_matrix[A.ID][B.ID] * (d_after_ornode + B.demand + 8) - \
                                        self.distance_matrix[B.ID][C.ID] * (d_after_ornode + 8)

                            originRtCostChange = cost_rt1

                            # cost_rt1_penalized += self.distance_matrix_penalized[A.ID][C.ID] * (d_after_ornode + 8) - \
                            #     self.distance_matrix_penalized[A.ID][B.ID] * (d_after_ornode + B.demand + 8) - \
                            #     self.distance_matrix_penalized[B.ID][C.ID] * (d_after_ornode + 8)

                            for i in range(0, targetNodeIndex):
                                nextnode = i + 1
                                n1 = rt2.sequenceOfNodes[i]
                                n2 = rt2.sequenceOfNodes[nextnode]
                                cost_rt2 += B.demand * self.distance_matrix[n1.ID][n2.ID]
                                # cost_rt2_penalized += B.demand * self.distance_matrix[n1.ID][n2.ID]

                            cost_rt2 += self.distance_matrix[F.ID][B.ID] * (d_after_tarnode + B.demand + 8) + \
                                        self.distance_matrix[B.ID][G.ID] * (d_after_tarnode + 8) - \
                                        self.distance_matrix[F.ID][G.ID] * (d_after_tarnode + 8)

                            targetRtCostChange = cost_rt2

                            # cost_rt2_penalized+= self.distance_matrix_penalized[F.ID][B.ID] * (d_after_tarnode + B.demand + 8) + \
                            #     self.distance_matrix_penalized[B.ID][G.ID] * (d_after_tarnode + 8) - \
                            #     self.distance_matrix_penalized[F.ID][G.ID] * (d_after_tarnode + 8)

                            moveCost = originRtCostChange + targetRtCostChange

                            # moveCost_penalized = cost_rt1_penalized + cost_rt2_penalized
                        # if (moveCost_penalized < rm.moveCost_penalized):
                        if (moveCost < rm.moveCost):
                            self.StoreBestRelocationMove(originRouteIndex, targetRouteIndex, originNodeIndex,
                                                         targetNodeIndex, moveCost, originRtCostChange,
                                                         targetRtCostChange, rm)

    def FindBestSwapMove(self, sm):
        heap = []

        for firstRouteIndex in range(0, len(self.sol.routes)):
            rt1: Route = self.sol.routes[firstRouteIndex]
            pr1 = rt1.cost
            for secondRouteIndex in range(firstRouteIndex, len(self.sol.routes)):
                rt2: Route = self.sol.routes[secondRouteIndex]
                pr2 = rt2.cost
                for firstNodeIndex in range(1, len(rt1.sequenceOfNodes) - 1):
                    startOfSecondNodeIndex = 1
                    if rt1 == rt2:
                        startOfSecondNodeIndex = firstNodeIndex + 1
                    for secondNodeIndex in range(startOfSecondNodeIndex, len(rt2.sequenceOfNodes) - 1):
                        b1 = rt1.sequenceOfNodes[firstNodeIndex]
                        b2 = rt2.sequenceOfNodes[secondNodeIndex]

                        if rt1 != rt2:
                            if rt1.load - b1.demand + b2.demand > self.capacity or \
                                    rt2.load - b2.demand + b1.demand > self.capacity:
                                continue

                            rt1.sequenceOfNodes[firstNodeIndex] = b2
                            rt2.sequenceOfNodes[secondNodeIndex] = b1
                            cost1, d1 = self.calculate_route_details(rt1.sequenceOfNodes, 6)
                            cost2, d2 = self.calculate_route_details(rt2.sequenceOfNodes, 6)
                            rt1.sequenceOfNodes[firstNodeIndex] = b1
                            rt2.sequenceOfNodes[secondNodeIndex] = b2
                            moveCost = - pr1 - pr2 + cost1 + cost2
                            costChangeFirstRoute = - pr1 + cost1
                            costChangeSecondRoute = - pr2 + cost2
                        else:
                            rt1.sequenceOfNodes[firstNodeIndex] = b2
                            rt1.sequenceOfNodes[secondNodeIndex] = b1
                            cost1, d1 = self.calculate_route_details(rt1.sequenceOfNodes, 6)
                            rt1.sequenceOfNodes[firstNodeIndex] = b1
                            rt1.sequenceOfNodes[secondNodeIndex] = b2
                            moveCost = - pr1 + cost1
                            costChangeFirstRoute = - pr1 + cost1
                            costChangeSecondRoute = costChangeFirstRoute

                        # Αποθήκευση πιθανής κίνησης στο heap
                        if moveCost < sm.moveCost and abs(moveCost) > 0.0001:
                            heapq.heappush(heap, (moveCost, firstRouteIndex, secondRouteIndex,
                                                  firstNodeIndex, secondNodeIndex,
                                                  costChangeFirstRoute, costChangeSecondRoute))

        if heap:
            bestMove = heapq.heappop(heap)
            self.StoreBestSwapMove(bestMove[1], bestMove[2], bestMove[3], bestMove[4],
                                   bestMove[0], bestMove[5], bestMove[6], sm)

        return sm

    def ApplyRelocationMove(self, rm: RelocationMove):

        # oldCost = self.CalculateTotalCost(self.sol)

        originRt = self.sol.routes[rm.originRoutePosition]
        targetRt = self.sol.routes[rm.targetRoutePosition]

        B = originRt.sequenceOfNodes[rm.originNodePosition]

        if originRt == targetRt:
            del originRt.sequenceOfNodes[rm.originNodePosition]
            if (rm.originNodePosition < rm.targetNodePosition):
                targetRt.sequenceOfNodes.insert(rm.targetNodePosition, B)
            else:
                targetRt.sequenceOfNodes.insert(rm.targetNodePosition + 1, B)

            originRt.cost += rm.moveCost
        else:
            del originRt.sequenceOfNodes[rm.originNodePosition]
            targetRt.sequenceOfNodes.insert(rm.targetNodePosition + 1, B)
            originRt.cost += rm.costChangeOriginRt
            targetRt.cost += rm.costChangeTargetRt
            originRt.load -= B.demand
            targetRt.load += B.demand

        self.sol.cost += rm.moveCost

        # newCost = self.CalculateTotalCost(self.sol)
        # debuggingOnly
        # if abs((newCost - oldCost) - rm.moveCost) > 0.0001:
        #     print('Cost Issue')

    def ApplySwapMove(self, sm):
        rt1 = self.sol.routes[sm.positionOfFirstRoute]
        rt2 = self.sol.routes[sm.positionOfSecondRoute]
        b1 = rt1.sequenceOfNodes[sm.positionOfFirstNode]
        b2 = rt2.sequenceOfNodes[sm.positionOfSecondNode]
        rt1.sequenceOfNodes[sm.positionOfFirstNode] = b2
        rt2.sequenceOfNodes[sm.positionOfSecondNode] = b1

        if rt1 == rt2:
            rt1.cost += sm.moveCost
        else:
            rt1.cost += sm.costChangeFirstRt
            rt2.cost += sm.costChangeSecondRt

            rt1.load = rt1.load - b1.demand + b2.demand
            rt2.load = rt2.load + b1.demand - b2.demand

        self.sol.cost += sm.moveCost
        # self.TestSolution()

    def ReportSolution(self, sol):
        total_tonne_kilometre = self.calc_tonne_kilometre_total(sol)
        for i in range(0, len(sol.routes)):
            rt = sol.routes[i]
            for j in range(0, len(rt.sequenceOfNodes) - 1):
                print(rt.sequenceOfNodes[j].ID, end=' ')
            print(f'Cost: {rt.cost} Load: {rt.load}')
        print(total_tonne_kilometre)

    def GetLastOpenRoute(self):
        if len(self.sol.routes) == 0:
            return None
        else:
            return self.sol.routes[-1]

    def IdentifyBestInsertion(self, bestInsertion, lastOpenRoute):
        """
            Identifies the best customer to insert into the last open route stochastically.
            Parameters:
                - lastOpenRoute: The last open route to which a customer will be inserted.
        """
        random.seed(1)
        candidateInsertions = []
        lastNodePresentInTheRoute = lastOpenRoute.sequenceOfNodes[-2]
        for i in range(0, len(self.customers)):
            candidateCust: Node = self.customers[i]
            if candidateCust.isRouted is False:
                if lastOpenRoute.load + candidateCust.demand <= lastOpenRoute.capacity:
                    trialCost = self.distance_matrix[lastNodePresentInTheRoute.ID][candidateCust.ID]
                    candidateInsertions.append((candidateCust, trialCost))
                else:
                    continue

        if not candidateInsertions:
            return

        # Extract costs and normalize to probabilities
        costs = [c[1] for c in candidateInsertions]
        max_cost = max(costs)
        exp_costs = [math.exp(-(cost - max_cost)) for cost in costs]
        total = sum(exp_costs)
        probabilities = [exp_cost / total for exp_cost in exp_costs]

        # Stochastically select a customer
        chosen_index = random.choices(range(len(candidateInsertions)), probabilities)[0]
        chosen_customer = candidateInsertions[chosen_index][0]

        # Update the bestInsertion object
        bestInsertion.customer = chosen_customer
        bestInsertion.route = lastOpenRoute
        bestInsertion.insertionPosition = len(lastOpenRoute.sequenceOfNodes) - 2
        bestInsertion.cost = candidateInsertions[chosen_index][1]

    def ApplyCustomerInsertion(self, insertion):
        insCustomer = insertion.customer
        rt = insertion.route
        # before the second depot occurrence
        insIndex = len(rt.sequenceOfNodes) - 1
        rt.sequenceOfNodes.insert(insIndex, insCustomer)

        beforeInserted = rt.sequenceOfNodes[-3]

        costAdded = self.distance_matrix[beforeInserted.ID][insCustomer.ID] + self.distance_matrix[insCustomer.ID][
            self.depot.ID]
        costRemoved = self.distance_matrix[beforeInserted.ID][self.depot.ID]

        rt.cost += costAdded - costRemoved
        self.sol.cost += costAdded - costRemoved

        rt.load += insCustomer.demand

        insCustomer.isRouted = True

    def StoreBestRelocationMove(self, originRouteIndex, targetRouteIndex, originNodeIndex, targetNodeIndex, moveCost,
                                originRtCostChange, targetRtCostChange, rm: RelocationMove):
        rm.originRoutePosition = originRouteIndex
        rm.originNodePosition = originNodeIndex
        rm.targetRoutePosition = targetRouteIndex
        rm.targetNodePosition = targetNodeIndex
        rm.costChangeOriginRt = originRtCostChange
        rm.costChangeTargetRt = targetRtCostChange
        rm.moveCost = moveCost
        # rm.moveCost_penalized = moveCost_penalized

    def StoreBestSwapMove(self, firstRouteIndex, secondRouteIndex, firstNodeIndex, secondNodeIndex, moveCost,
                          costChangeFirstRoute, costChangeSecondRoute, sm):
        sm.positionOfFirstRoute = firstRouteIndex
        sm.positionOfSecondRoute = secondRouteIndex
        sm.positionOfFirstNode = firstNodeIndex
        sm.positionOfSecondNode = secondNodeIndex
        sm.costChangeFirstRt = costChangeFirstRoute
        sm.costChangeSecondRt = costChangeSecondRoute
        sm.moveCost = moveCost
        # sm.moveCost_penalized = moveCost_penalized

    # Allagh
    def CalculateTotalCost(self, sol):
        c = 0
        for i in range(0, len(sol.routes)):
            rt = sol.routes[i]
            for j in range(0, len(rt.sequenceOfNodes) - 1):
                a = rt.sequenceOfNodes[j]
                b = rt.sequenceOfNodes[j + 1]
                c += self.distance_matrix[a.ID][b.ID]
        return c

    def InitializeOperators(self, rm, sm, top):
        rm.Initialize()
        sm.Initialize()
        top.Initialize()

    def FindBestTwoOptMove(self, top, cost_of_segment_to_be=None):
        for rtInd1 in range(0, len(self.sol.routes)):
            rt1: Route = self.sol.routes[rtInd1]
            for rtInd2 in range(rtInd1, len(self.sol.routes)):
                rt2: Route = self.sol.routes[rtInd2]
                for nodeInd1 in range(0, len(rt1.sequenceOfNodes) - 1):
                    start2 = 0
                    if (rt1 == rt2):
                        start2 = nodeInd1 + 2

                    for nodeInd2 in range(start2, len(rt2.sequenceOfNodes) - 1):

                        A = rt1.sequenceOfNodes[nodeInd1]
                        B = rt1.sequenceOfNodes[nodeInd1 + 1]
                        K = rt2.sequenceOfNodes[nodeInd2]
                        L = rt2.sequenceOfNodes[nodeInd2 + 1]

                        if rt1 == rt2:
                            if nodeInd1 == 0 and nodeInd2 == len(rt1.sequenceOfNodes) - 2:
                                continue

                            rd1 = self.CalcRD(rt1, nodeInd1)
                            rd2 = self.CalcRD(rt2, nodeInd2)

                            extrac1 = 0
                            current_load = rd2 + B.demand
                            for i in range(nodeInd1 + 1, nodeInd2):
                                n1 = rt1.sequenceOfNodes[i]
                                n2 = rt1.sequenceOfNodes[i + 1]
                                extrac1 += current_load * self.distance_matrix[n1.ID][n2.ID]
                                current_load += n2.demand

                            current_load = rd2 + K.demand
                            for i in range(nodeInd2, nodeInd1 + 1, -1):
                                n2 = rt1.sequenceOfNodes[i]
                                n1 = rt1.sequenceOfNodes[i - 1]
                                extrac1 -= current_load * self.distance_matrix[n1.ID][n2.ID]
                                current_load += n1.demand

                            cost = 0

                            cost += self.distance_matrix[A.ID][K.ID] * (rd1) + \
                                    self.distance_matrix[B.ID][L.ID] * (rd2) - \
                                    self.distance_matrix[A.ID][B.ID] * (rd1) - \
                                    self.distance_matrix[K.ID][L.ID] * (rd2)
                            moveCost = cost + extrac1
                        else:
                            if nodeInd1 == 0 and nodeInd2 == 0:
                                continue
                            if nodeInd1 == len(rt1.sequenceOfNodes) - 2 and nodeInd2 == len(rt2.sequenceOfNodes) - 2:
                                continue

                            rd1 = self.CalcRD(rt1, nodeInd1 - 1)
                            rd2 = self.CalcRD(rt2, nodeInd2 - 1)

                            if self.CapacityIsViolated(rt1, nodeInd1, rt2, nodeInd2, rd1, rd2):
                                continue

                            cost_cummulative_change_rt1 = 0
                            for i in range(0, nodeInd1):
                                next = i + 1
                                n1 = rt1.sequenceOfNodes[i]
                                n2 = rt1.sequenceOfNodes[next]
                                cost_cummulative_change_rt1 += rd2 * self.distance_matrix[n1.ID][n2.ID] - \
                                                               rd1 * self.distance_matrix[n1.ID][n2.ID]

                            cost_cummulative_change_rt2 = 0
                            for i in range(0, nodeInd2):
                                next = i + 1
                                n1 = rt2.sequenceOfNodes[i]
                                n2 = rt2.sequenceOfNodes[next]
                                cost_cummulative_change_rt2 += rd1 * self.distance_matrix[n1.ID][n2.ID] - \
                                                               rd2 * self.distance_matrix[n1.ID][n2.ID]

                            costAdded = self.distance_matrix[A.ID][L.ID] * rd2 + \
                                        self.distance_matrix[K.ID][B.ID] * rd1
                            costRemoved = self.distance_matrix[A.ID][B.ID] * rd1 + \
                                          self.distance_matrix[K.ID][L.ID] * rd2

                            moveCost = costAdded - costRemoved + \
                                       cost_cummulative_change_rt1 + cost_cummulative_change_rt2

                        if moveCost < top.moveCost and abs(moveCost) > 0.0001:
                            self.StoreBestTwoOptMove(rtInd1, rtInd2, nodeInd1, nodeInd2, moveCost, top)

    def CalcRD(self, rt, nodeInd):
        rd = 8
        for i in range(len(rt.sequenceOfNodes) - 1, nodeInd - 1, -1):
            rd = rd + rt.sequenceOfNodes[i].demand

        return rd

    def CapacityIsViolated(self, rt1, nodeInd1, rt2, nodeInd2, rd1, rd2):

        rt1FirstSegmentLoad = rt1.load - rd1 + rt1.sequenceOfNodes[nodeInd1].demand
        rt1SecondSegmentLoad = rd1 - rt1.sequenceOfNodes[nodeInd1].demand

        rt2FirstSegmentLoad = rt2.load - rd2 + rt2.sequenceOfNodes[nodeInd2].demand
        rt2SecondSegmentLoad = rd2 - rt2.sequenceOfNodes[nodeInd2].demand

        if rt1FirstSegmentLoad + rt2SecondSegmentLoad > rt1.capacity:
            return True
        if rt2FirstSegmentLoad + rt1SecondSegmentLoad > rt2.capacity:
            return True

        return False

    def StoreBestTwoOptMove(self, rtInd1, rtInd2, nodeInd1, nodeInd2, moveCost, top):
        top.positionOfFirstRoute = rtInd1
        top.positionOfSecondRoute = rtInd2
        top.positionOfFirstNode = nodeInd1
        top.positionOfSecondNode = nodeInd2
        top.moveCost = moveCost

    def ApplyTwoOptMove(self, top):
        rt1: Route = self.sol.routes[top.positionOfFirstRoute]
        rt2: Route = self.sol.routes[top.positionOfSecondRoute]

        if rt1 == rt2:
            # reverses the nodes in the segment [positionOfFirstNode + 1,  top.positionOfSecondNode]
            reversedSegment = reversed(rt1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1])
            # lst = list(reversedSegment)
            # lst2 = list(reversedSegment)
            rt1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1] = reversedSegment

            # reversedSegmentList = list(reversed(rt1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1]))
            # rt1.sequenceOfNodes[top.positionOfFirstNode + 1: top.positionOfSecondNode + 1] = reversedSegmentList

            rt1.cost += top.moveCost

        else:
            # slice with the nodes from position top.positionOfFirstNode + 1 onwards
            relocatedSegmentOfRt1 = rt1.sequenceOfNodes[top.positionOfFirstNode + 1:]

            # slice with the nodes from position top.positionOfFirstNode + 1 onwards
            relocatedSegmentOfRt2 = rt2.sequenceOfNodes[top.positionOfSecondNode + 1:]

            del rt1.sequenceOfNodes[top.positionOfFirstNode + 1:]
            del rt2.sequenceOfNodes[top.positionOfSecondNode + 1:]

            rt1.sequenceOfNodes.extend(relocatedSegmentOfRt2)
            rt2.sequenceOfNodes.extend(relocatedSegmentOfRt1)

            self.UpdateRouteCostAndLoad(rt1)
            self.UpdateRouteCostAndLoad(rt2)

        self.sol.cost += top.moveCost

    def UpdateRouteCostAndLoad(self, rt: Route):
        tot_dem = sum(n.demand for n in rt.sequenceOfNodes)
        rt.load = tot_dem
        tot_load = 8 + rt.load
        tn_km = 0
        for i in range(len(rt.sequenceOfNodes) - 1):
            from_node = rt.sequenceOfNodes[i]
            to_node = rt.sequenceOfNodes[i + 1]
            tn_km += self.distance_matrix[from_node.ID][to_node.ID] * tot_load
            tot_load -= to_node.demand
        rt.cost = tn_km

    def TestSolution(self):
        totalSolCost = 0
        for r in range(0, len(self.sol.routes)):
            rt: Route = self.sol.routes[r]
            rtCost = 0
            rtLoad = 0
            for n in range(0, len(rt.sequenceOfNodes) - 1):
                A = rt.sequenceOfNodes[n]
                B = rt.sequenceOfNodes[n + 1]
                rtCost += self.distance_matrix[A.ID][B.ID]
                rtLoad += A.demand
            if abs(rtCost - rt.cost) > 0.0001:
                print('Route Cost problem')
            if rtLoad != rt.load:
                print('Route Load problem')

            totalSolCost += rt.cost

        if abs(totalSolCost - self.sol.cost) > 0.0001:
            print('Solution Cost problem')

    def IdentifyBestInsertionAllPositions(self, bestInsertion, rt):
        for i in range(0, len(self.customers)):
            candidateCust: Node = self.customers[i]
            if candidateCust.isRouted is False:
                if rt.load + candidateCust.demand <= rt.capacity:
                    lastNodePresentInTheRoute = rt.sequenceOfNodes[-2]
                    for j in range(0, len(rt.sequenceOfNodes) - 1):
                        A = rt.sequenceOfNodes[j]
                        B = rt.sequenceOfNodes[j + 1]
                        costAdded = self.distance_matrix[A.ID][candidateCust.ID] + \
                                    self.distance_matrix[candidateCust.ID][
                                        B.ID]
                        costRemoved = self.distance_matrix[A.ID][B.ID]
                        trialCost = costAdded - costRemoved

                        if trialCost < bestInsertion.cost:
                            bestInsertion.customer = candidateCust
                            bestInsertion.route = rt
                            bestInsertion.cost = trialCost
                            bestInsertion.insertionPosition = j

    def ApplyCustomerInsertionAllPositions(self, insertion):
        insCustomer = insertion.customer
        rt = insertion.route
        # before the second depot occurrence
        insIndex = insertion.insertionPosition
        rt.sequenceOfNodes.insert(insIndex + 1, insCustomer)
        rt.cost += insertion.cost
        self.sol.cost += insertion.cost
        rt.load += insCustomer.demand
        insCustomer.isRouted = True

    def penalize_arcs(self):
        # if self.penalized_n1_ID != -1 and self.penalized_n2_ID != -1:
        #     self.distance_matrix_penalized[self.penalized_n1_ID][self.penalized_n2_ID] = self.distance_matrix[self.penalized_n1_ID][self.penalized_n2_ID]
        #     self.distance_matrix_penalized[self.penalized_n2_ID][self.penalized_n1_ID] = self.distance_matrix[self.penalized_n2_ID][self.penalized_n1_ID]
        max_criterion = 0
        pen_1 = -1
        pen_2 = -1
        for i in range(len(self.sol.routes)):
            rt = self.sol.routes[i]
            for j in range(len(rt.sequenceOfNodes) - 1):
                id1 = rt.sequenceOfNodes[j].ID
                id2 = rt.sequenceOfNodes[j + 1].ID
                criterion = self.distance_matrix[id1][id2] / (1 + self.times_penalized[id1][id2])
                if criterion > max_criterion:
                    max_criterion = criterion
                    pen_1 = id1
                    pen_2 = id2
        self.times_penalized[pen_1][pen_2] += 1
        self.times_penalized[pen_2][pen_1] += 1

        pen_weight = 0.15

        self.distance_matrix_penalized[pen_1][pen_2] = (1 + pen_weight * self.times_penalized[pen_1][pen_2]) * \
                                                       self.distance_matrix[pen_1][pen_2]
        self.distance_matrix_penalized[pen_2][pen_1] = (1 + pen_weight * self.times_penalized[pen_2][pen_1]) * \
                                                       self.distance_matrix[pen_2][pen_1]
        self.penalized_n1_ID = pen_1
        self.penalized_n2_ID = pen_2


def write_to_file(solution, output_file):
    with open(output_file, 'w') as file:
        file.write(f'Cost:\n{solution.cost}\n')
        file.write('Routes:\n')
        file.write(str(len(solution.routes)) + '\n')
        for route in solution.routes:
            route_str = '0'
            for node in route.sequenceOfNodes[
                        1:len(route.sequenceOfNodes) - 1]:  # The first node in each route is 0 and should not be repeated
                route_str += ',' + str(node.ID)
            file.write(route_str + '\n')
