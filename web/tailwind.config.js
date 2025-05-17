/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#F5F7FA', // snežno bela
          100: '#E8EDF2', // jutranja megla
          200: '#D1DBE6',
          300: '#B0C4D8',
          400: '#8FABC9',
          500: '#526D82', // zimsko nebo
          600: '#435A6B',
          700: '#354754',
          800: '#26343D',
          900: '#172026'
        },
        secondary: {
          50: '#F2F5F5',
          100: '#E1E6E6',
          200: '#C9D1D1',
          300: '#ACB8B8',
          400: '#8F9F9F',
          500: '#728686',
          600: '#596D6D',
          700: '#405454',
          800: '#273B3B',
          900: '#0E2222'
        },
        accent: {
          50: '#F5F2F2',
          100: '#E6E1E1',
          200: '#D1C9C9',
          300: '#B8ACAC',
          400: '#9F8F8F',
          500: '#867272',
          600: '#6D5959',
          700: '#544040',
          800: '#3B2727',
          900: '#220E0E'
        },
        success: {
          50: '#F2F9F5',
          100: '#E5F3EB',
          200: '#CCE7D7',
          300: '#99D0B0',
          400: '#66B888',
          500: '#5D7285', // umirjena sivka
          600: '#4A5B6A',
          700: '#384450',
          800: '#252D35',
          900: '#13171B'
        },
        warning: {
          50: '#FEF8EE',
          100: '#FDF1DD',
          200: '#FBE3BB',
          300: '#F7C778',
          400: '#F4AB34',
          500: '#DDA15E', // zahod sonca
          600: '#B1814B',
          700: '#856138',
          800: '#584025',
          900: '#2C2013'
        },
        error: {
          50: '#FCF2F2',
          100: '#F9E5E5',
          200: '#F3CCCC',
          300: '#E79999',
          400: '#DB6666',
          500: '#BC4749', // skandinavska rdeča
          600: '#96393A',
          700: '#712B2C',
          800: '#4B1D1D',
          900: '#260E0F'
        },
        dark: {
          background: '#172026', // temno zimsko nebo
          card: '#26343D', // temna modrina
          border: '#354754', // temna modrina
          text: '#F5F7FA', // snežno bela
          subtext: '#9DB2BF', // ledeni odtenek
        }
      },
      fontFamily: {
        sans: [
          'Inter',
          'system-ui',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Roboto',
          'sans-serif'
        ],
        mono: [
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Monaco',
          'Consolas',
          'Liberation Mono',
          'Courier New',
          'monospace'
        ],
      },
      fontSize: {
        'xs': ['0.75rem', { lineHeight: '1rem' }],
        'sm': ['0.875rem', { lineHeight: '1.25rem' }],
        'base': ['1rem', { lineHeight: '1.5rem' }],
        'lg': ['1.125rem', { lineHeight: '1.75rem' }],
        'xl': ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
        '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
      },
      letterSpacing: {
        tighter: '-0.05em',
        tight: '-0.025em',
        normal: '0',
        wide: '0.025em',
        wider: '0.05em',
        widest: '0.1em',
      },
      boxShadow: {
        'sm': '0 2px 4px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.1)',
        'md': '0 4px 6px rgba(0, 0, 0, 0.05), 0 2px 4px rgba(0, 0, 0, 0.1)',
        'lg': '0 10px 15px rgba(0, 0, 0, 0.05), 0 4px 6px rgba(0, 0, 0, 0.1)',
        'xl': '0 20px 25px rgba(0, 0, 0, 0.05), 0 10px 10px rgba(0, 0, 0, 0.1)',
        'card': '0 2px 4px rgba(0, 0, 0, 0.02)',
        'card-hover': '0 4px 6px rgba(0, 0, 0, 0.04)',
        'card-dark': '0 2px 4px rgba(0, 0, 0, 0.1)',
        'card-dark-hover': '0 4px 6px rgba(0, 0, 0, 0.15)',
      },
      transitionDuration: {
        '150': '150ms',
        '300': '300ms',
        '500': '500ms',
      },
      transitionTimingFunction: {
        'ease-in-out': 'cubic-bezier(0.4, 0, 0.2, 1)',
        'ease-out': 'cubic-bezier(0, 0, 0.2, 1)',
        'ease-in': 'cubic-bezier(0.4, 0, 1, 1)',
      },
    },
  },
  plugins: [],
};