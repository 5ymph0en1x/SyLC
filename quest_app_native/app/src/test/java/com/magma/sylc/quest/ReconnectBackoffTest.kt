package com.magma.sylc.quest

import org.junit.Assert.assertEquals
import org.junit.Test

class ReconnectBackoffTest {

    private fun backoff() = ReconnectBackoff(
        initialDelayMs = 250L,
        maxDelayMs = 3_000L,
        stableConnectionNs = 5_000_000_000L,
    )

    @Test
    fun `first retry waits the initial delay`() {
        assertEquals(250L, backoff().onSessionEnded(livedNs = 0L))
    }

    @Test
    fun `repeated immediate failures double the wait up to the ceiling`() {
        val policy = backoff()
        val waits = (1..8).map { policy.onSessionEnded(livedNs = 1_000L) }

        assertEquals(listOf(250L, 500L, 1_000L, 2_000L, 3_000L, 3_000L, 3_000L, 3_000L), waits)
    }

    @Test
    fun `a connection that lived long enough clears the penalty`() {
        val policy = backoff()
        repeat(4) { policy.onSessionEnded(livedNs = 1_000L) }   // climb to the ceiling

        // A session that stayed up past the stability threshold is not evidence
        // of a broken link, so the next drop retries immediately.
        assertEquals(250L, policy.onSessionEnded(livedNs = 30_000_000_000L))
        assertEquals(250L, policy.onSessionEnded(livedNs = 1_000L))
    }

    @Test
    fun `a session exactly at the threshold counts as stable`() {
        val policy = backoff()
        policy.onSessionEnded(livedNs = 1_000L)

        assertEquals(250L, policy.onSessionEnded(livedNs = 5_000_000_000L))
    }

    @Test
    fun `reset returns to the initial delay`() {
        val policy = backoff()
        repeat(3) { policy.onSessionEnded(livedNs = 1_000L) }
        policy.reset()

        assertEquals(250L, policy.delayMs)
        assertEquals(250L, policy.onSessionEnded(livedNs = 1_000L))
    }
}
