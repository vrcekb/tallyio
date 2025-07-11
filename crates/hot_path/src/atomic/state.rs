//! Atomic state machines for lock-free state management.
//!
//! This module provides atomic state machine implementations for managing
//! complex state transitions in a lock-free manner.

use alloc::string::String;
use core::sync::atomic::{AtomicU32, Ordering};

/// State transition result
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum StateTransition {
    /// Transition was successful
    Success,
    /// Transition failed due to invalid state
    InvalidState,
    /// Transition failed due to concurrent modification
    ConcurrentModification,
    /// Transition failed for other reason
    Failed(String),
}

impl core::fmt::Display for StateTransition {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Success => return write!(f, "State transition successful"),
            Self::InvalidState => return write!(f, "Invalid state for transition"),
            Self::ConcurrentModification => return write!(f, "Concurrent modification detected"),
            Self::Failed(msg) => return write!(f, "State transition failed: {msg}"),
        }
    }
}

/// Atomic state machine for lock-free state management
#[repr(C, align(64))]
#[non_exhaustive]
pub struct AtomicStateMachine {
    /// Current state value
    state: AtomicU32,
    /// State transition counter
    transition_count: AtomicU32,
    /// Padding for cache alignment
    padding: [u8; 56],
}

impl AtomicStateMachine {
    /// Create a new atomic state machine
    #[must_use]
    #[inline]
    pub const fn new(initial_state: u32) -> Self {
        return Self {
            state: AtomicU32::new(initial_state),
            transition_count: AtomicU32::new(0),
            padding: [0; 56],
        };
    }

    /// Get the current state
    #[must_use]
    #[inline]
    pub fn get_state(&self) -> u32 {
        return self.state.load(Ordering::Acquire);
    }

    /// Attempt to transition from one state to another
    #[inline]
    pub fn transition(&self, from_state: u32, to_state: u32) -> StateTransition {
        match self.state.compare_exchange_weak(
            from_state,
            to_state,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                self.transition_count.fetch_add(1, Ordering::Relaxed);
                return StateTransition::Success;
            }
            Err(current) => {
                if current == from_state {
                    return StateTransition::ConcurrentModification;
                }
                return StateTransition::InvalidState;
            }
        }
    }

    /// Force set the state (use with caution)
    #[inline]
    pub fn force_set_state(&self, new_state: u32) {
        self.state.store(new_state, Ordering::Release);
        self.transition_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the number of transitions that have occurred
    #[must_use]
    #[inline]
    pub fn get_transition_count(&self) -> u32 {
        return self.transition_count.load(Ordering::Relaxed);
    }

    /// Check if the state machine is in a specific state
    #[must_use]
    #[inline]
    pub fn is_in_state(&self, expected_state: u32) -> bool {
        return self.get_state() == expected_state;
    }

    /// Wait for a specific state (busy wait)
    #[inline]
    pub fn wait_for_state(&self, expected_state: u32) {
        while !self.is_in_state(expected_state) {
            core::hint::spin_loop();
        }
    }

    /// Try to transition with retry logic
    #[inline]
    pub fn transition_with_retry(&self, from_state: u32, to_state: u32, max_retries: u32) -> StateTransition {
        for _ in 0..max_retries {
            match self.transition(from_state, to_state) {
                StateTransition::Success => return StateTransition::Success,
                StateTransition::ConcurrentModification => {
                    core::hint::spin_loop();
                }
                other @ (StateTransition::InvalidState | StateTransition::Failed(_)) => return other,
            }
        }
        return StateTransition::Failed("Max retries exceeded".into());
    }
}

impl Default for AtomicStateMachine {
    #[inline]
    fn default() -> Self {
        return Self::new(0);
    }
}

// Common state constants
/// Idle state constant
pub const STATE_IDLE: u32 = 0;
/// Running state constant
pub const STATE_RUNNING: u32 = 1;
/// Stopping state constant
pub const STATE_STOPPING: u32 = 2;
/// Stopped state constant
pub const STATE_STOPPED: u32 = 3;
/// Error state constant
pub const STATE_ERROR: u32 = 4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_machine_creation() {
        let sm = AtomicStateMachine::new(STATE_IDLE);
        assert_eq!(sm.get_state(), STATE_IDLE);
        assert_eq!(sm.get_transition_count(), 0);
    }

    #[test]
    fn test_successful_transition() {
        let sm = AtomicStateMachine::new(STATE_IDLE);
        let result = sm.transition(STATE_IDLE, STATE_RUNNING);
        assert_eq!(result, StateTransition::Success);
        assert_eq!(sm.get_state(), STATE_RUNNING);
        assert_eq!(sm.get_transition_count(), 1);
    }

    #[test]
    fn test_invalid_transition() {
        let sm = AtomicStateMachine::new(STATE_IDLE);
        let result = sm.transition(STATE_RUNNING, STATE_STOPPED);
        assert_eq!(result, StateTransition::InvalidState);
        assert_eq!(sm.get_state(), STATE_IDLE);
    }

    #[test]
    fn test_force_set_state() {
        let sm = AtomicStateMachine::new(STATE_IDLE);
        sm.force_set_state(STATE_ERROR);
        assert_eq!(sm.get_state(), STATE_ERROR);
        assert_eq!(sm.get_transition_count(), 1);
    }

    #[test]
    fn test_is_in_state() {
        let sm = AtomicStateMachine::new(STATE_IDLE);
        assert!(sm.is_in_state(STATE_IDLE));
        assert!(!sm.is_in_state(STATE_RUNNING));
    }

    #[test]
    fn test_transition_with_retry() {
        let sm = AtomicStateMachine::new(STATE_IDLE);
        let result = sm.transition_with_retry(STATE_IDLE, STATE_RUNNING, 3);
        assert_eq!(result, StateTransition::Success);
        assert_eq!(sm.get_state(), STATE_RUNNING);
    }
}
