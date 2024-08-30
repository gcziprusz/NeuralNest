# Fundamental Concepts in Probability

1. **Sample Space**  
   **Definition**: The set of all possible outcomes of a random experiment. It's usually denoted by \( S \).  
   **Example**: For a six-sided die, the sample space is \( S = \{1, 2, 3, 4, 5, 6\} \).

2. **Event**  
   **Definition**: A subset of the sample space. An event can consist of one or more outcomes.  
   **Example**: Rolling an even number on a die is an event, \( A = \{2, 4, 6\} \).

3. **Probability of an Event**  
   **Definition**: A measure of the likelihood that an event will occur, ranging from 0 (impossible) to 1 (certain).  
   **Formula**: For a finite sample space with equally likely outcomes, 
   \[
   P(A) = \frac{|A|}{|S|}
   \]
   where \( |A| \) is the number of favorable outcomes and \( |S| \) is the total number of outcomes.  
   **Example**: The probability of rolling a 3 on a six-sided die is \( P(\{3\}) = \frac{1}{6} \).

4. **Complementary Rule**  
   **Definition**: The probability of the complement of an event \( A \) (i.e., the event that \( A \) does not occur) is \( 1 \) minus the probability of \( A \).  
   **Formula**: 
   \[
   P(A^c) = 1 - P(A)
   \]
   **Example**: If the probability of raining tomorrow is \( 0.3 \), the probability of not raining is \( 0.7 \).

5. **Sum Rule (Addition Rule)**  
   **Definition**: Used to calculate the probability of the union of two events. There are different forms depending on whether the events are mutually exclusive or not.  
   **Formulas**:  
   - Mutually Exclusive: 
   \[
   P(A \cup B) = P(A) + P(B)
   \]
   - Non-Mutually Exclusive: 
   \[
   P(A \cup B) = P(A) + P(B) - P(A \cap B)
   \]

6. **Product Rule (Multiplication Rule)**  
   **Definition**: Used to calculate the probability of the intersection of two events.  
   **Formulas**:  
   - Independent Events: 
   \[
   P(A \cap B) = P(A) \times P(B)
   \]
   - Dependent Events: 
   \[
   P(A \cap B) = P(A) \times P(B|A)
   \]
   where \( P(B|A) \) is the conditional probability of \( B \) given \( A \) has occurred.

7. **Conditional Probability**  
   **Definition**: The probability of one event occurring given that another event has already occurred.  
   **Formula**: 
   \[
   P(B|A) = \frac{P(A \cap B)}{P(A)}
   \]
   provided \( P(A) > 0 \).

8. **Law of Total Probability**  
   **Definition**: A way to calculate the probability of an event by considering all possible ways in which it can occur, particularly when dealing with conditional probabilities.  
   **Formula**: If \( B_1, B_2, \dots, B_n \) are mutually exclusive and exhaustive events, then:
   \[
   P(A) = P(A \cap B_1) + P(A \cap B_2) + \dots + P(A \cap B_n)
   \]
   Or equivalently:
   \[
   P(A) = P(B_1)P(A|B_1) + P(B_2)P(A|B_2) + \dots + P(B_n)P(A|B_n)
   \]

9. **Bayes' Theorem**  
   **Definition**: A fundamental theorem that relates conditional probabilities. It allows the computation of \( P(A|B) \) from \( P(B|A) \).  
   **Formula**: 
   \[
   P(A|B) = \frac{P(B|A)P(A)}{P(B)}
   \]

10. **Independence**  
    **Definition**: Two events \( A \) and \( B \) are independent if the occurrence of one does not affect the occurrence of the other.  
    **Formula**: If \( A \) and \( B \) are independent, the probability of both events occurring is the product of their individual probabilities: 
    \[
    P(A \cap B) = P(A) \times P(B)
    \]
    **Example**: If you flip a coin and roll a six-sided die, the events of getting heads (event \( A \)) and rolling a 4 (event \( B \)) are independent because the outcome of one does not influence the outcome of the other. Therefore, the probability of both events occurring is:
    \[
    P(A \cap B) = P(\text{Heads}) \times P(\text{Rolling a 4}) = \frac{1}{2} \times \frac{1}{6} = \frac{1}{12}
    \]

### Summary:
- **Sample Space**: The set of all possible outcomes.
- **Event**: A subset of the sample space.
- **Probability of an Event**: \( P(A) = \frac{|A|}{|S|} \).
- **Complementary Rule**: \( P(A^c) = 1 - P(A) \).
- **Sum Rule (Addition Rule)**: \( P(A \cup B) = P(A) + P(B) \) (if mutually exclusive); \( P(A \cup B) = P(A) + P(B) - P(A \cap B) \) (if not).
- **Product Rule (Multiplication Rule)**: \( P(A \cap B) = P(A) \times P(B) \) (if independent); \( P(A \cap B) = P(A) \times P(B|A) \) (if dependent).
- **Conditional Probability**: \( P(B|A) = \frac{P(A \cap B)}{P(A)} \).
- **Law of Total Probability**: \( P(A) = \sum_{i=1}^{n} P(B_i)P(A|B_i) \).
- **Bayes' Theorem**: \( P(A|B) = \frac{P(B|A)P(A)}{P(B)} \).
- **Independence**: \( P(A \cap B) = P(A) \times P(B) \).