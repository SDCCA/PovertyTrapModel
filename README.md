![image](https://github.com/SDCCA/PovertyTrapModel/assets/112457375/51aea7d8-46b4-4716-b683-169a02d22ec6)# Summary of changes to Main branch:

April 5, 2023 - Made sigma (risk averseness) heterogeneous among agents between an arbitrary range from 0.9 to 1.9; the original value was 1.5.

April 10, 2023 - Generalized functions income_generation, isocline, and income_updation and calculations for self.front(s) and self.slope to work with any number of tech options housed in a dictionary, TechTable.

May 31, 2023 - Changed the consumption calculation method to optimization with scipy. 
             - Added agent attribute theta, which is a personal perception of future shocks. This value is updated at every time step based on their sensitivity to observed theta.

July 28, 2023 - Added adaptation options housed in AdapTable from which agents can pick based on value maximization.
              - Adjusted give_money() function so that it updates and is reflected in capital, newly named trade_money(). Agents also now trade with a portion of their entire capital, not just a portion of capital generated in that timestep.

September 20, 2023 - Adjusted trade_money() (formerly give_money()) so that agents have an opportunity to trade with all of their neighbors. 
                   - Introduced automatic output file naming and command-line argument options.

