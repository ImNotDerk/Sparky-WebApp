'use client';
import React from 'react';

const LandingPage: React.FC = () => {
  const handleStartChatting = () => {
    window.location.href = '/chat';
  };

  return (
    <main
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        padding: '1.5rem',
        color: 'white',
        background: 'linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%)'
      }}
    >
      <div
        style={{
          textAlign: 'center',
          backgroundColor: 'rgba(31, 41, 55, 0.5)',
          backdropFilter: 'blur(10px)',
          padding: '2.5rem',
          borderRadius: '1rem',
          boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.18)',
          maxWidth: '42rem',
          width: '100%'
        }}
      >

        {/* Chatbot Header */}
        <h1
          style={{
            fontSize: '3.75rem',
            lineHeight: '1',
            fontWeight: 700,
            color: '#93C5FD',
            marginBottom: '1rem',
            textShadow: '2px 2px 4px rgba(0,0,0,0.5)'
          }}
        >
          🧠 Welcome to SPARKY
        </h1>

        {/* Subheading/Description */}
        <p
          style={{
            fontSize: '1.25rem',
            lineHeight: '1.75rem',
            color: '#E5E7EB',
            marginBottom: '2rem'
          }}
        >
          Your friendly Grade 3 peer tutor, ready to help you learn and explore!
        </p>

        {/* Call to Action Button */}
        <button
          onClick={handleStartChatting}
          style={{
            backgroundColor: '#3B82F6',
            color: 'white',
            fontWeight: 'bold',
            fontSize: '1.25rem',
            padding: '1rem 2rem',
            borderRadius: '9999px',
            border: 'none',
            cursor: 'pointer',
            boxShadow: '0 4px 14px 0 rgba(0, 118, 255, 0.39)',
            transition: 'transform 0.3s ease',
          }}
          onMouseOver={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
          onMouseOut={(e) => e.currentTarget.style.transform = 'scale(1)'}
        >
          Start Chatting Now 🚀
        </button>
      </div>

      {/* Footer */}
      <footer
        style={{
          position: 'absolute',
          bottom: '1rem',
          color: '#9CA3AF',
          fontSize: '0.875rem'
        }}
      >
        <p>Created with fun and learning in mind.</p>
      </footer>
    </main>
  );
};

export default LandingPage;

