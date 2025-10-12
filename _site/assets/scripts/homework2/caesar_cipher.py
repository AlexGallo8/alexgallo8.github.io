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

def caesar_encrypt(text, shift):
    """
    Encrypts text using Caesar cipher with given shift.
    Only encrypts letters, preserves case and other characters.
    """
    result = []
    for char in text:
        if char.isalpha():
            # Determine if uppercase or lowercase
            base = ord('A') if char.isupper() else ord('a')
            # Apply shift with wraparound
            shifted = (ord(char) - base + shift) % 26
            result.append(chr(base + shifted))
        else:
            result.append(char)
    return ''.join(result)

def analyze_frequency(text):
    """
    Analyzes letter frequency in text.
    Returns dictionary of letter: percentage and total letter count.
    """
    # Convert to lowercase and keep only letters
    letters = [c.lower() for c in text if c.isalpha()]
    total = len(letters)
    
    if total == 0:
        return {}, 0
    
    # Count occurrences
    counts = Counter(letters)
    
    # Convert to percentages
    frequencies = {letter: (count / total) * 100 
                   for letter, count in counts.items()}
    
    # Ensure all letters a-z are present (even if 0%)
    for letter in string.ascii_lowercase:
        if letter not in frequencies:
            frequencies[letter] = 0.0
    
    return dict(sorted(frequencies.items())), total

def chi_squared_score(observed_freq, expected_freq):
    """
    Calculates chi-squared statistic between observed and expected frequencies.
    Lower score = better match.
    """
    score = 0
    for letter in string.ascii_lowercase:
        expected = expected_freq.get(letter, 0.1)  # Small value to avoid division by zero
        observed = observed_freq.get(letter, 0)
        score += ((observed - expected) ** 2) / expected
    return score

def decrypt_with_frequency_analysis(encrypted_text, language='english'):
    """
    Decrypts Caesar cipher using frequency analysis.
    Tries all 26 possible shifts and returns the one with best match to language.
    """
    expected_freq = LANGUAGE_FREQUENCIES[language]
    best_shift = 0
    best_score = float('inf')
    results = []
    
    # Try all possible shifts
    for shift in range(26):
        decrypted = caesar_encrypt(encrypted_text, shift)
        observed_freq, _ = analyze_frequency(decrypted)
        score = chi_squared_score(observed_freq, expected_freq)
        
        results.append({
            'shift': shift,
            'score': score,
            'decrypted_preview': decrypted[:100]
        })
        
        if score < best_score:
            best_score = score
            best_shift = shift
    
    best_decryption = caesar_encrypt(encrypted_text, best_shift)
    
    return {
        'best_shift': best_shift,
        'original_shift': (26 - best_shift) % 26,
        'decrypted_text': best_decryption,
        'chi_squared_score': best_score,
        'all_attempts': sorted(results, key=lambda x: x['score'])
    }

def plot_frequency_comparison(original_freq, encrypted_freq, decrypted_freq, 
                              expected_freq, language, output_dir='assets/images'):
    """
    Creates comparison plots of letter frequencies.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    letters = list(string.ascii_lowercase)
    
    # Prepare data for plotting
    orig_values = [original_freq.get(l, 0) for l in letters]
    enc_values = [encrypted_freq.get(l, 0) for l in letters]
    dec_values = [decrypted_freq.get(l, 0) for l in letters]
    exp_values = [expected_freq.get(l, 0) for l in letters]
    
    # Plot 1: Original vs Expected
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(letters))
    width = 0.35
    ax.bar([i - width/2 for i in x], orig_values, width, label='Original Text', alpha=0.8)
    ax.bar([i + width/2 for i in x], exp_values, width, label=f'Expected ({language.title()})', alpha=0.8)
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
    
    # Plot 2: Encrypted frequency
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(letters, enc_values, color='coral', alpha=0.8)
    ax.set_xlabel('Letters')
    ax.set_ylabel('Frequency (%)')
    ax.set_title('Encrypted Text Letter Distribution')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/encrypted_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Encrypted vs Expected (for visual shift detection)
    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.35
    ax.bar([i - width/2 for i in x], enc_values, width, label='Encrypted Text', alpha=0.8, color='coral')
    ax.bar([i + width/2 for i in x], exp_values, width, label=f'Expected ({language.title()})', alpha=0.8, color='steelblue')
    ax.set_xlabel('Letters')
    ax.set_ylabel('Frequency (%)')
    ax.set_title('Encrypted vs Expected Distribution - Visual Shift Detection')
    ax.set_xticks(x)
    ax.set_xticklabels(letters)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/encrypted_vs_expected.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: All distributions comparison
    fig, ax = plt.subplots(figsize=(16, 8))
    width = 0.2
    x = range(len(letters))
    ax.bar([i - 1.5*width for i in x], orig_values, width, label='Original', alpha=0.8)
    ax.bar([i - 0.5*width for i in x], enc_values, width, label='Encrypted', alpha=0.8)
    ax.bar([i + 0.5*width for i in x], dec_values, width, label='Decrypted', alpha=0.8)
    ax.bar([i + 1.5*width for i in x], exp_values, width, label=f'Expected ({language.title()})', alpha=0.8)
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

def save_frequency_table(frequencies, filename):
    """
    Saves frequency data to CSV file.
    """
    df = pd.DataFrame([
        {'letter': letter, 'frequency_%': freq}
        for letter, freq in sorted(frequencies.items())
    ])
    df.to_csv(filename, index=False)
    print(f"Frequency table saved to {filename}")

def main():
    # Step 1: Read or input text
    print("=== Caesar Cipher Frequency Analysis ===\n")
    
    # Option 1: Read from file
    with open('assets/files/input_text.txt', 'r', encoding='utf-8') as f:
        original_text = f.read()
    
    # Option 2: Use sample text (replace with your chosen text)
    # original_text = """
    # Your chosen text goes here. This should be a substantial text
    # in the language you want to analyze (English, Italian, etc.).
    # The longer the text, the better the statistical analysis will be.
    # Aim for at least 500-1000 characters for good results.
    # """
    
    # Choose language
    language = 'english'  # or 'italian'
    
    # Step 2: Generate random shift and encrypt
    original_shift = random.randint(1, 25)
    print(f"Random shift applied: {original_shift}")
    encrypted_text = caesar_encrypt(original_text, original_shift)
    
    # Save encrypted text
    with open('assets/files/encrypted_text.txt', 'w', encoding='utf-8') as f:
        f.write(encrypted_text)
    print("Encrypted text saved to 'assets/files/encrypted_text.txt'\n")
    
    # Step 3: Analyze frequencies
    print("Analyzing frequencies...")
    original_freq, orig_count = analyze_frequency(original_text)
    encrypted_freq, enc_count = analyze_frequency(encrypted_text)
    
    print(f"Total letters in original: {orig_count}")
    print(f"Total letters in encrypted: {enc_count}\n")
    
    # Save frequency tables
    os.makedirs('assets/data', exist_ok=True)
    save_frequency_table(original_freq, 'assets/data/original_frequencies.csv')
    save_frequency_table(encrypted_freq, 'assets/data/encrypted_frequencies.csv')
    
    # Step 4: Decrypt using frequency analysis
    print("Attempting decryption using frequency analysis...")
    result = decrypt_with_frequency_analysis(encrypted_text, language)
    
    print(f"\n=== RESULTS ===")
    print(f"Original shift used: {original_shift}")
    print(f"Detected shift: {result['original_shift']}")
    print(f"Chi-squared score: {result['chi_squared_score']:.2f}")
    print(f"Success: {'YES' if result['original_shift'] == original_shift else 'NO'}")
    
    # Analyze decrypted text
    decrypted_freq, dec_count = analyze_frequency(result['decrypted_text'])
    save_frequency_table(decrypted_freq, 'assets/data/decrypted_frequencies.csv')
    
    # Save decrypted text
    with open('assets/files/decrypted_text.txt', 'w', encoding='utf-8') as f:
        f.write(result['decrypted_text'])
    print("\nDecrypted text saved to 'assets/files/decrypted_text.txt'")
    
    # Step 5: Create visualizations
    print("\nGenerating plots...")
    expected_freq = LANGUAGE_FREQUENCIES[language]
    plot_frequency_comparison(original_freq, encrypted_freq, decrypted_freq, 
                             expected_freq, language)
    print("Plots saved to 'assets/images/'")
    
    # Show top 5 attempts
    print("\n=== Top 5 Decryption Attempts (by chi-squared score) ===")
    for i, attempt in enumerate(result['all_attempts'][:5], 1):
        original = (26 - attempt['shift']) % 26
        print(f"\n{i}. Decrypt shift: {attempt['shift']}, Original shift: {original}, Score: {attempt['score']:.2f}")
        print(f"   Preview: {attempt['decrypted_preview']}...")

if __name__ == "__main__":
    main()