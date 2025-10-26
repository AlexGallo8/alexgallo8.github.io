import string
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import os

# Expected letter frequencies for different languages (in percentages)
LANGUAGE_FREQUENCIES = {
    # Source: Peter Norvig's analysis of Google Books corpus (2013)
    # http://norvig.com/mayzner.html
    # Based on analysis of 3.5 trillion characters from Google Books
    'english': {
        'e': 12.49, 't': 9.28, 'a': 8.04, 'o': 7.64, 'i': 7.57, 'n': 7.23,
        's': 6.51, 'r': 6.28, 'h': 5.05, 'l': 4.07, 'd': 3.82, 'c': 3.34,
        'u': 2.73, 'm': 2.51, 'f': 2.40, 'p': 2.14, 'g': 1.87, 'w': 1.68,
        'y': 1.66, 'b': 1.48, 'v': 1.05, 'k': 0.54, 'x': 0.23, 'j': 0.16,
        'q': 0.12, 'z': 0.09
    },
    # Source: Wikipedia - Letter Frequency
    # https://en.wikipedia.org/wiki/Letter_frequency
    # Based on analysis of Italian text corpora
    'italian': {
        'e': 11.79, 'a': 11.74, 'i': 11.28, 'o': 9.83, 'n': 6.88, 'l': 6.51,
        'r': 6.37, 't': 5.62, 's': 4.98, 'c': 4.50, 'd': 3.73, 'p': 3.05,
        'u': 3.01, 'm': 2.51, 'v': 2.10, 'g': 1.64, 'h': 1.54, 'f': 0.95,
        'b': 0.92, 'q': 0.51, 'z': 0.49, 'j': 0.00, 'k': 0.00, 'w': 0.00,
        'x': 0.00, 'y': 0.00
    }
}

# Common bigrams in English (from norvig.com/mayzner.html)
# Source: Analysis of Google Books corpus
COMMON_BIGRAMS = ['TH', 'HE', 'IN', 'ER', 'AN', 'RE', 'ON', 'AT', 'EN', 'ND', 
                  'TI', 'ES', 'OR', 'TE', 'OF', 'ED', 'IS', 'IT', 'AL', 'AR',
                  'ST', 'TO', 'NT', 'NG', 'SE', 'HA', 'AS', 'OU', 'IO', 'LE',
                  'VE', 'CO', 'ME', 'DE', 'HI', 'RI', 'RO', 'IC', 'NE', 'EA',
                  'RA', 'CE', 'LI', 'CH', 'LL', 'BE', 'MA', 'SI', 'OM', 'UR']

# Common trigrams in English (from norvig.com/mayzner.html)
# Top 50 most frequent trigrams
COMMON_TRIGRAMS = ['THE', 'AND', 'ING', 'HER', 'HAT', 'HIS', 'THA', 'ERE', 'FOR', 'ENT',
                   'ION', 'TER', 'WAS', 'YOU', 'ITH', 'VER', 'ALL', 'WIT', 'THI', 'TIO',
                   'HER', 'EST', 'HIS', 'OFT', 'ITH', 'FTH', 'STH', 'OTH', 'RES', 'ONT',
                   'EAR', 'TIN', 'EDI', 'ATE', 'IST', 'RAT', 'ERS', 'OUR', 'HEN', 'INT',
                   'SAN', 'EEN', 'ARE', 'REA', 'VEN', 'STA', 'DTH', 'COM', 'MAN', 'OUR']

# ============= RSA FUNCTIONS =============

def gcd(a, b):
    """Calculate Greatest Common Divisor"""
    while b != 0:
        a, b = b, a % b
    return a

def mod_inverse(e, phi):
    """Calculate modular inverse using Extended Euclidean Algorithm"""
    m0, x0, x1 = phi, 0, 1
    if phi == 1:
        return 0
    while e > 1:
        q = e // phi
        phi, e = e % phi, phi
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1

def mod_pow(base, exp, mod):
    """Fast modular exponentiation"""
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result

def generate_rsa_keys(p=61, q=53, e=17):
    """
    Generate RSA keys with given primes.
    Using small primes for educational purposes - INSECURE!
    """
    n = p * q
    phi = (p - 1) * (q - 1)
    d = mod_inverse(e, phi)
    return {
        'p': p,
        'q': q,
        'n': n,
        'e': e,
        'd': d,
        'phi': phi
    }

def rsa_encrypt_char(char, e, n):
    """Encrypt a single character using RSA"""
    m = ord(char)
    return mod_pow(m, e, n)

def rsa_decrypt_char(cipher_num, d, n):
    """Decrypt a single number using RSA"""
    m = mod_pow(cipher_num, d, n)
    return chr(m)

def rsa_encrypt_text(text, e, n):
    """
    Encrypt text letter-by-letter using RSA.
    This is INSECURE and vulnerable to frequency analysis!
    """
    encrypted = []
    for char in text:
        if char.isalpha():
            encrypted.append(rsa_encrypt_char(char.upper(), e, n))
        else:
            encrypted.append(None)  # Preserve spaces and punctuation
    return encrypted

def rsa_decrypt_text(encrypted_numbers, d, n):
    """Decrypt RSA encrypted numbers back to text"""
    decrypted = []
    for num in encrypted_numbers:
        if num is not None:
            decrypted.append(rsa_decrypt_char(num, d, n))
        else:
            decrypted.append(' ')
    return ''.join(decrypted)

# ============= FREQUENCY ANALYSIS FUNCTIONS =============

def analyze_frequency(text):
    """
    Analyzes letter frequency in text.
    Returns dictionary of letter: percentage and total letter count.
    """
    letters = [c.upper() for c in text if c.isalpha()]
    total = len(letters)
    
    if total == 0:
        return {}, 0
    
    counts = Counter(letters)
    frequencies = {letter: (count / total) * 100 
                   for letter, count in counts.items()}
    
    # Ensure all letters A-Z are present
    for letter in string.ascii_uppercase:
        if letter not in frequencies:
            frequencies[letter] = 0.0
    
    return dict(sorted(frequencies.items())), total

def analyze_number_frequency(encrypted_numbers):
    """
    Analyzes frequency of encrypted numbers.
    Returns dictionary of number: percentage and total count.
    """
    numbers = [n for n in encrypted_numbers if n is not None]
    total = len(numbers)
    
    if total == 0:
        return {}, 0
    
    counts = Counter(numbers)
    frequencies = {num: (count / total) * 100 
                   for num, count in counts.items()}
    
    return dict(sorted(frequencies.items(), key=lambda x: x[1], reverse=True)), total

def average_distance_score(observed_freq, expected_freq, sorted_letters):
    """
    Calculate average distance between observed and expected frequencies.
    This is the professor's suggested metric.
    """
    total_distance = 0
    count = 0
    
    for i, letter in enumerate(sorted_letters):
        if i < len(sorted_letters):
            obs = observed_freq.get(letter, 0)
            exp = expected_freq.get(letter.lower(), 0)
            total_distance += abs(obs - exp)
            count += 1
    
    return total_distance / count if count > 0 else float('inf')

def chi_squared_score(observed_freq, expected_freq):
    """
    Calculates chi-squared statistic between observed and expected frequencies.
    Lower score = better match.
    """
    score = 0
    for letter in string.ascii_uppercase:
        expected = expected_freq.get(letter.lower(), 0.1)
        observed = observed_freq.get(letter, 0)
        if expected > 0:
            score += ((observed - expected) ** 2) / expected
    return score

def count_ngrams(text, n=2):
    """Count occurrences of n-grams in text"""
    ngrams = Counter()
    text_upper = text.upper()
    for i in range(len(text_upper) - n + 1):
        ngram = text_upper[i:i+n]
        # Only count if all characters are letters
        if all(c.isalpha() for c in ngram) and len(ngram) == n:
            ngrams[ngram] += 1
    return ngrams

def score_bigrams(text):
    """Score text based on common English bigrams"""
    bigrams = count_ngrams(text, n=2)
    score = sum(bigrams.get(bg, 0) for bg in COMMON_BIGRAMS)
    return score

def score_trigrams(text):
    """Score text based on common English trigrams"""
    trigrams = count_ngrams(text, n=3)
    score = sum(trigrams.get(tg, 0) for tg in COMMON_TRIGRAMS)
    return score

def score_combined(text, bigram_weight=1.0, trigram_weight=2.0):
    """
    Combined scoring using both bigrams and trigrams.
    Trigrams have higher weight as they're more discriminative.
    """
    bi_score = score_bigrams(text)
    tri_score = score_trigrams(text)
    return bigram_weight * bi_score + trigram_weight * tri_score

# ============= FREQUENCY ATTACK FUNCTIONS =============

def create_initial_mapping(encrypted_numbers, language='english'):
    """
    Create initial mapping based on frequency analysis.
    Maps most frequent number to most frequent letter, etc.
    """
    # Get frequency of encrypted numbers
    num_freq, _ = analyze_number_frequency(encrypted_numbers)
    
    # Sort numbers by frequency (descending)
    sorted_numbers = sorted(num_freq.keys(), key=lambda x: num_freq[x], reverse=True)
    
    # Get expected letter frequencies
    expected_freq = LANGUAGE_FREQUENCIES[language]
    sorted_letters = sorted(expected_freq.keys(), key=lambda x: expected_freq[x], reverse=True)
    sorted_letters = [l.upper() for l in sorted_letters]
    
    # Create mapping: most frequent number -> most frequent letter
    mapping = {}
    for i, num in enumerate(sorted_numbers):
        if i < len(sorted_letters):
            mapping[num] = sorted_letters[i]
    
    return mapping, num_freq, sorted_letters

def decrypt_with_mapping(encrypted_numbers, mapping):
    """Decrypt encrypted numbers using a letter mapping"""
    decrypted = []
    for num in encrypted_numbers:
        if num is not None:
            decrypted.append(mapping.get(num, '?'))
        else:
            decrypted.append(' ')
    return ''.join(decrypted)

def optimize_rare_letters(encrypted_numbers, initial_mapping, 
                         max_iterations=100, bottom_n=12):
    """
    Optimize mapping focusing on RARE/INFREQUENT letters.
    
    After bigram optimization, common letters (E, T, A, O, I, N, S, H, R) are usually correct.
    The remaining errors are in rare letters (J, K, Q, X, Z, V, W, B, G, Y, P, F).
    
    Strategy: Try swapping the LEAST frequent letters and score with trigrams.
    """
    import random
    
    best_mapping = initial_mapping.copy()
    best_score = score_trigrams(decrypt_with_mapping(encrypted_numbers, best_mapping))
    
    attempts = [{
        'iteration': 0,
        'score': best_score,
        'mapping': best_mapping.copy(),
        'preview': decrypt_with_mapping(encrypted_numbers, best_mapping)[:100],
        'type': 'rare_letters'
    }]
    
    # Get the LEAST frequent numbers (these map to rare letters)
    all_numbers = list(initial_mapping.keys())
    rare_numbers = all_numbers[-bottom_n:] if len(all_numbers) > bottom_n else all_numbers
    
    print(f"   Focusing on {len(rare_numbers)} least frequent letters...")
    print(f"   Initial score: {best_score}")
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try swapping pairs of rare letters
        for i in range(len(rare_numbers)):
            for j in range(i + 1, len(rare_numbers)):
                # Create test mapping with swapped letters
                test_mapping = best_mapping.copy()
                test_mapping[rare_numbers[i]], test_mapping[rare_numbers[j]] = \
                    test_mapping[rare_numbers[j]], test_mapping[rare_numbers[i]]
                
                # Score with trigrams
                test_score = score_trigrams(decrypt_with_mapping(encrypted_numbers, test_mapping))
                
                if test_score > best_score:
                    best_score = test_score
                    best_mapping = test_mapping
                    improved = True
                    
                    print(f"   Iteration {iteration}: Score improved to {best_score} "
                          f"(swapped rare letters {test_mapping[rare_numbers[j]]}↔{test_mapping[rare_numbers[i]]})")
                    
                    attempts.append({
                        'iteration': iteration,
                        'score': best_score,
                        'mapping': best_mapping.copy(),
                        'preview': decrypt_with_mapping(encrypted_numbers, best_mapping)[:100],
                        'type': 'rare_letters'
                    })
                    break
            if improved:
                break
    
    print(f"   Rare letter optimization: {len(attempts)-1} improvements made")
    
    return best_mapping, best_score, attempts

def optimize_mapping_with_ngrams(encrypted_numbers, initial_mapping, 
                                 max_iterations=100, top_n=10,
                                 use_trigrams=False, use_annealing=False):
    """
    Optimize mapping using hill-climbing with n-gram scoring.
    
    HOW IT WORKS:
    1. Start with initial mapping (based on letter frequencies)
    2. Calculate initial score (count of common bigrams/trigrams)
    3. Try swapping pairs of letters in the mapping
    4. If swap improves score, keep it
    5. Repeat until no improvement found
    
    For trigrams, uses simulated annealing to escape local maxima.
    
    Args:
        encrypted_numbers: List of encrypted numbers
        initial_mapping: Initial number->letter mapping
        max_iterations: Maximum number of iterations
        top_n: How many of the most frequent numbers to consider for swapping
        use_trigrams: If True, use trigrams; if False, use bigrams
        use_annealing: If True, use simulated annealing (accepts worse solutions sometimes)
    """
    import math
    import random
    
    best_mapping = initial_mapping.copy()
    current_mapping = initial_mapping.copy()
    
    # Choose scoring function
    if use_trigrams:
        score_func = score_trigrams
        ngram_type = "trigrams"
    else:
        score_func = score_bigrams
        ngram_type = "bigrams"
    
    best_score = score_func(decrypt_with_mapping(encrypted_numbers, best_mapping))
    current_score = best_score
    
    attempts = [{
        'iteration': 0,
        'score': best_score,
        'mapping': best_mapping.copy(),
        'preview': decrypt_with_mapping(encrypted_numbers, best_mapping)[:100],
        'type': ngram_type
    }]
    
    improved = True
    iteration = 0
    
    # Simulated annealing parameters
    temperature = 100.0
    cooling_rate = 0.95
    
    print(f"   Starting {'simulated annealing' if use_annealing else 'hill-climbing'} with {ngram_type}...")
    print(f"   Initial score: {best_score}")
    print(f"   Search space: top {top_n} letters")
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Get numbers to consider - more for trigrams
        numbers = list(current_mapping.keys())[:min(top_n, len(current_mapping))]
        
        # Shuffle to explore different pairs
        if use_annealing:
            random.shuffle(numbers)
        
        # Try swapping pairs
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                # Create test mapping with swapped letters
                test_mapping = current_mapping.copy()
                test_mapping[numbers[i]], test_mapping[numbers[j]] = \
                    test_mapping[numbers[j]], test_mapping[numbers[i]]
                
                # Score this mapping
                test_score = score_func(decrypt_with_mapping(encrypted_numbers, test_mapping))
                
                # Decide whether to accept this swap
                accept = False
                
                if use_annealing:
                    # Simulated annealing: accept worse solutions with probability
                    if test_score > current_score:
                        accept = True
                    else:
                        # Accept worse solution with probability based on temperature
                        delta = test_score - current_score
                        probability = math.exp(delta / temperature)
                        if random.random() < probability:
                            accept = True
                            print(f"   Iteration {iteration}: Accepted worse solution (score {test_score}) "
                                  f"at temperature {temperature:.1f}")
                else:
                    # Standard hill-climbing: only accept improvements
                    if test_score > current_score:
                        accept = True
                
                if accept:
                    current_score = test_score
                    current_mapping = test_mapping
                    improved = True
                    
                    # Update best if this is the best so far
                    if test_score > best_score:
                        best_score = test_score
                        best_mapping = test_mapping.copy()
                        
                        print(f"   Iteration {iteration}: Score improved to {best_score} "
                              f"(swapped {test_mapping[numbers[j]]}↔{test_mapping[numbers[i]]})")
                        
                        attempts.append({
                            'iteration': iteration,
                            'score': best_score,
                            'mapping': best_mapping.copy(),
                            'preview': decrypt_with_mapping(encrypted_numbers, best_mapping)[:100],
                            'type': ngram_type
                        })
                    break
            if improved:
                break
        
        # Cool down temperature for annealing
        if use_annealing:
            temperature *= cooling_rate
    
    print(f"   Optimization completed: {len(attempts)-1} improvements made")
    
    return best_mapping, best_score, attempts

def frequency_attack(encrypted_numbers, language='english', optimize=True):
    """
    Complete frequency analysis attack on RSA encrypted text.
    Uses a two-phase approach:
    1. Initial mapping based on letter frequencies
    2. Optimization using bigrams, then trigrams
    """
    print("Phase 1: Creating initial mapping based on letter frequencies...")
    initial_mapping, num_freq, sorted_letters = create_initial_mapping(
        encrypted_numbers, language
    )
    
    # Decrypt with initial mapping
    initial_decrypted = decrypt_with_mapping(encrypted_numbers, initial_mapping)
    initial_freq, _ = analyze_frequency(initial_decrypted)
    
    # Calculate scores
    expected_freq = LANGUAGE_FREQUENCIES[language]
    avg_dist = average_distance_score(initial_freq, expected_freq, sorted_letters)
    chi_sq = chi_squared_score(initial_freq, expected_freq)
    bigram_score = score_bigrams(initial_decrypted)
    trigram_score = score_trigrams(initial_decrypted)
    
    print(f"   Average distance score: {avg_dist:.3f}")
    print(f"   Chi-squared score: {chi_sq:.2f}")
    print(f"   Initial bigram score: {bigram_score}")
    print(f"   Initial trigram score: {trigram_score}")
    
    result = {
        'initial_mapping': initial_mapping,
        'initial_decrypted': initial_decrypted,
        'initial_freq': initial_freq,
        'avg_distance': avg_dist,
        'chi_squared': chi_sq,
        'initial_bigram_score': bigram_score,
        'initial_trigram_score': trigram_score,
        'number_frequencies': num_freq,
        'sorted_letters': sorted_letters
    }
    
    if optimize:
        # Phase 2: Optimize with bigrams
        print("\nPhase 2: Optimizing mapping using bigram analysis...")
        bigram_mapping, bigram_opt_score, bigram_attempts = optimize_mapping_with_ngrams(
            encrypted_numbers, initial_mapping, 
            use_trigrams=False, use_annealing=False, top_n=10
        )
        
        bigram_decrypted = decrypt_with_mapping(encrypted_numbers, bigram_mapping)
        
        print(f"   Final bigram score: {bigram_opt_score}")
        print(f"   Improvement: {bigram_opt_score - bigram_score} "
              f"(+{((bigram_opt_score/bigram_score - 1) * 100):.1f}%)")
        
        # Phase 3: Further optimize with trigrams using simulated annealing
        print("\nPhase 3: Further optimizing using trigram analysis with simulated annealing...")
        trigram_mapping, trigram_opt_score, trigram_attempts = optimize_mapping_with_ngrams(
            encrypted_numbers, bigram_mapping, 
            use_trigrams=True, use_annealing=True, 
            top_n=15,  # Search more letters for trigrams
            max_iterations=150  # More iterations for annealing
        )
        
        trigram_decrypted = decrypt_with_mapping(encrypted_numbers, trigram_mapping)
        current_trigram = score_trigrams(bigram_decrypted)
        
        print(f"   Final trigram score: {trigram_opt_score}")
        print(f"   Improvement: {trigram_opt_score - current_trigram} "
              f"(+{((trigram_opt_score/max(current_trigram, 1) - 1) * 100):.1f}%)")
        
        result.update({
            'bigram_mapping': bigram_mapping,
            'bigram_decrypted': bigram_decrypted,
            'bigram_score': bigram_opt_score,
            'bigram_attempts': bigram_attempts,
            'trigram_mapping': trigram_mapping,
            'trigram_decrypted': trigram_decrypted,
            'trigram_score': trigram_opt_score,
            'trigram_attempts': trigram_attempts,
            'final_mapping': trigram_mapping,
            'final_decrypted': trigram_decrypted
        })
    
    return result

# ============= VISUALIZATION FUNCTIONS =============

def plot_frequency_comparison(original_freq, encrypted_num_freq, decrypted_freq,
                              expected_freq, language, output_dir='assets/images'):
    """Create comparison plots of letter frequencies"""
    os.makedirs(output_dir, exist_ok=True)
    
    letters = list(string.ascii_uppercase)
    
    # Prepare data
    orig_values = [original_freq.get(l, 0) for l in letters]
    dec_values = [decrypted_freq.get(l, 0) for l in letters]
    exp_values = [expected_freq.get(l.lower(), 0) for l in letters]
    
    # Plot 1: Original vs Expected
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(letters))
    width = 0.35
    ax.bar([i - width/2 for i in x], orig_values, width, label='Original Text', alpha=0.8)
    ax.bar([i + width/2 for i in x], exp_values, width, 
           label=f'Expected ({language.title()})', alpha=0.8)
    ax.set_xlabel('Letters')
    ax.set_ylabel('Frequency (%)')
    ax.set_title('Original Text vs Expected Language Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(letters)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/original_vs_expected.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Number frequency distribution
    fig, ax = plt.subplots(figsize=(14, 6))
    sorted_nums = sorted(encrypted_num_freq.items(), key=lambda x: x[1], reverse=True)[:26]
    nums = [str(n) for n, _ in sorted_nums]
    freqs = [f for _, f in sorted_nums]
    ax.bar(nums, freqs, color='coral', alpha=0.8)
    ax.set_xlabel('Encrypted Numbers')
    ax.set_ylabel('Frequency (%)')
    ax.set_title('Encrypted Numbers Distribution (Top 26)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/encrypted_numbers_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: All distributions comparison
    fig, ax = plt.subplots(figsize=(16, 8))
    width = 0.25
    x = range(len(letters))
    ax.bar([i - width for i in x], orig_values, width, label='Original', alpha=0.8)
    ax.bar(x, dec_values, width, label='Decrypted', alpha=0.8)
    ax.bar([i + width for i in x], exp_values, width, 
           label=f'Expected ({language.title()})', alpha=0.8)
    ax.set_xlabel('Letters')
    ax.set_ylabel('Frequency (%)')
    ax.set_title('Complete Frequency Analysis Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(letters)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/complete_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_mapping_table(mapping, num_freq, filename):
    """Save the number-to-letter mapping as CSV"""
    data = []
    for num, letter in sorted(mapping.items(), key=lambda x: num_freq.get(x[0], 0), reverse=True):
        data.append({
            'encrypted_number': num,
            'mapped_letter': letter,
            'frequency_%': num_freq.get(num, 0)
        })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Mapping table saved to {filename}")

# ============= MAIN FUNCTION =============

def main():
    print("="*60)
    print("RSA Letter-by-Letter Frequency Analysis Attack")
    print("with Bigram and Trigram Optimization")
    print("="*60)
    
    # Step 1: Read input text
    print("\nStep 1: Reading input text...")
    with open('assets/files/input_text.txt', 'r', encoding='utf-8') as f:
        original_text = f.read()
    
    # Clean text (only uppercase letters and spaces)
    original_text_clean = ''.join(c if c.isalpha() or c.isspace() else '' 
                                   for c in original_text.upper())
    print(f"Text length: {len(original_text_clean)} characters")
    
    # Choose language
    language = 'english'
    
    # Step 2: Generate RSA keys and encrypt
    print("\nStep 2: Generating RSA keys and encrypting...")
    keys = generate_rsa_keys()
    print(f"RSA Parameters: p={keys['p']}, q={keys['q']}, n={keys['n']}, e={keys['e']}")
    print(f"⚠️  Private key d={keys['d']} (hidden from attacker)")
    
    encrypted_numbers = rsa_encrypt_text(original_text_clean, keys['e'], keys['n'])
    
    # Save encrypted data
    os.makedirs('assets/files', exist_ok=True)
    with open('assets/files/encrypted_numbers.txt', 'w') as f:
        f.write(','.join(str(n) if n is not None else 'SPACE' for n in encrypted_numbers))
    print("Encrypted numbers saved to 'assets/files/encrypted_numbers.txt'")
    
    # Step 3: Analyze frequencies
    print("\nStep 3: Analyzing frequencies...")
    original_freq, orig_count = analyze_frequency(original_text_clean)
    print(f"Total letters: {orig_count}")
    
    os.makedirs('assets/data', exist_ok=True)
    pd.DataFrame([{'letter': k, 'frequency_%': v} for k, v in original_freq.items()]).to_csv(
        'assets/data/original_frequencies.csv', index=False
    )
    
    # Step 4: Frequency attack!
    print("\n" + "="*60)
    print("FREQUENCY ATTACK (without knowing private key!)")
    print("="*60)
    
    result = frequency_attack(encrypted_numbers, language=language, optimize=True)
    
    # Step 5: Compare with actual decryption
    print("\n" + "="*60)
    print("VERIFICATION (using private key)")
    print("="*60)
    actual_decrypted = rsa_decrypt_text(encrypted_numbers, keys['d'], keys['n'])
    
    # Calculate accuracy for each phase
    initial = result['initial_decrypted']
    bigram_opt = result.get('bigram_decrypted', initial)
    trigram_opt = result.get('final_decrypted', bigram_opt)
    
    def calc_accuracy(text1, text2):
        correct = sum(1 for a, b in zip(text1, text2) if a == b and a != ' ')
        total = sum(1 for c in text1 if c != ' ')
        return (correct / total * 100) if total > 0 else 0, correct, total
    
    acc_init, corr_init, total = calc_accuracy(actual_decrypted, initial)
    acc_bi, corr_bi, _ = calc_accuracy(actual_decrypted, bigram_opt)
    acc_tri, corr_tri, _ = calc_accuracy(actual_decrypted, trigram_opt)
    
    print(f"\nAccuracy progression:")
    print(f"  Initial (frequency only):  {acc_init:.1f}% ({corr_init}/{total} correct)")
    print(f"  After bigram optimization: {acc_bi:.1f}% ({corr_bi}/{total} correct) [+{acc_bi-acc_init:.1f}%]")
    print(f"  After trigram optimization: {acc_tri:.1f}% ({corr_tri}/{total} correct) [+{acc_tri-acc_bi:.1f}%]")
    
    # Save results
    with open('assets/files/decrypted_initial.txt', 'w', encoding='utf-8') as f:
        f.write(initial)
    
    with open('assets/files/decrypted_bigram.txt', 'w', encoding='utf-8') as f:
        f.write(bigram_opt)
    
    with open('assets/files/decrypted_final.txt', 'w', encoding='utf-8') as f:
        f.write(trigram_opt)
    
    with open('assets/files/actual_decrypted.txt', 'w', encoding='utf-8') as f:
        f.write(actual_decrypted)
    
    print("\nAll decrypted versions saved to 'assets/files/'")
    
    # Save mappings
    save_mapping_table(result['initial_mapping'], result['number_frequencies'], 
                      'assets/data/mapping_initial.csv')
    save_mapping_table(result.get('bigram_mapping', result['initial_mapping']), 
                      result['number_frequencies'], 
                      'assets/data/mapping_bigram.csv')
    save_mapping_table(result.get('final_mapping', result['initial_mapping']), 
                      result['number_frequencies'], 
                      'assets/data/mapping_final.csv')
    
    # Step 6: Visualizations
    print("\nGenerating visualizations...")
    decrypted_freq, _ = analyze_frequency(trigram_opt)
    plot_frequency_comparison(
        original_freq, 
        result['number_frequencies'],
        decrypted_freq,
        LANGUAGE_FREQUENCIES[language],
        language
    )
    print("Plots saved to 'assets/images/'")
    
    # Step 7: Show results preview
    print("\n" + "="*60)
    print("RESULTS PREVIEW")
    print("="*60)
    print(f"\nOriginal text (first 200 chars):")
    print(actual_decrypted[:200])
    print(f"\nInitial decryption (frequency only, first 200 chars):")
    print(initial[:200])
    print(f"\nAfter bigram optimization (first 200 chars):")
    print(bigram_opt[:200])
    print(f"\nFinal decryption after trigram optimization (first 200 chars):")
    print(trigram_opt[:200])
    
    # Show optimization progress
    print(f"\n" + "="*60)
    print("OPTIMIZATION PROGRESS SUMMARY")
    print("="*60)
    
    if 'bigram_attempts' in result:
        bigram_attempts = result['bigram_attempts']
        print(f"\nBigram optimization: {len(bigram_attempts)-1} improvements")
        for i, attempt in enumerate(bigram_attempts[:3]):
            print(f"  Iteration {attempt['iteration']}: Score = {attempt['score']}")
    
    if 'trigram_attempts' in result:
        trigram_attempts = result['trigram_attempts']
        print(f"\nTrigram optimization: {len(trigram_attempts)-1} improvements")
        for i, attempt in enumerate(trigram_attempts[:3]):
            print(f"  Iteration {attempt['iteration']}: Score = {attempt['score']}")
    
    print("\n" + "="*60)
    print("Attack completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()